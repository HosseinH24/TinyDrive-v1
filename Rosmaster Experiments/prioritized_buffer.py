import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


class PrioritizedReplayBuffer:
    """
    A buffer for prioritized experience replay in VLM training.
    Inspired by PER in reinforcement learning but adapted for supervised learning.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_annealing: float = 0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of samples to store
            alpha: Priority exponent (α) - how much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (β) - corrects bias from priority sampling (0 = no correction, 1 = full correction)
            beta_annealing: Rate at which beta increases to 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        
        # Storage for experiences
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Small constant to avoid zero priority
        self.epsilon = 1e-5
        
    def add(self, sample, priority=None):
        """
        Add a sample to the buffer with an initial priority.
        If no priority is given, assign max priority.
        """
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
            
        if self.size < self.capacity:
            self.buffer.append(sample)
            self.size += 1
        else:
            self.buffer[self.position] = sample
            
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch according to priorities.
        Returns samples, indices, and importance sampling weights.
        """
        if self.size < batch_size:
            indices = np.random.choice(self.size, size=batch_size, replace=True)
        else:
            # Create probability distribution based on priorities
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize to max weight
        weights = torch.FloatTensor(weights)
        
        samples = [self.buffer[idx] for idx in indices]
        
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for the given indices based on TD error or loss.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
            
    def is_full(self):
        """Check if buffer is at capacity"""
        return self.size >= self.capacity


class PrioritizedVQADataset(Dataset):
    """
    Dataset that uses prioritized sampling for VQA data
    """
    def __init__(self, token_entries, tokenizer, device, capacity=10000, alpha=0.6, beta=0.4):
        self.tokenizer = tokenizer
        self.device = device
        self.buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha, beta=beta)
        
        # Initialize buffer with all samples with default priority
        print("Initializing prioritized buffer with samples...")
        for entry in token_entries:
            emb = entry['combined_embeddings'].cpu()
            for i, ans_ids in enumerate(entry['answer_tokens']):
                question = entry['questions'][i]
                q_tok = self.tokenizer(
                    question,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=100
                ).input_ids.squeeze(0).cpu()
                
                sample = {
                    'combined_embedding': emb[i],
                    'labels': ans_ids.squeeze(0).cpu(),
                    'question_input_ids': q_tok,
                    'raw_data': {
                        'image_path': entry['image_path'],
                        'question': question
                    }
                }
                self.buffer.add(sample)
        
        print(f"Initialized buffer with {self.buffer.size} samples")

    def __len__(self):
        return self.buffer.size

    def __getitem__(self, idx):
        # This won't be used directly - we use PrioritizedSampler instead
        return self.buffer.buffer[idx]
    
    def update_priorities(self, indices, priorities):
        """Update sample priorities based on losses"""
        self.buffer.update_priorities(indices, priorities)


class PrioritizedSampler(Sampler):
    """
    Custom sampler that samples based on priorities in the dataset's buffer
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Sample according to priorities and return indices
        samples, indices, weights = self.dataset.buffer.sample(self.batch_size)
        return iter(indices)

    def __len__(self):
        return self.dataset.buffer.size


def compute_informativeness_score(
    logits, 
    labels, 
    combined_embeddings=None, 
    model_outputs=None,
    loss_weight=0.5,
    uncertainty_weight=0.3,
    diversity_weight=0.2,
    confidence_threshold=0.7,
    global_priorities=None
):
    """
    Improved scoring function with more balanced components:
    1. Loss value (weighted)
    2. Prediction uncertainty (weighted)
    3. Diversity bonus (weighted)
    4. Optional temporal component
    
    Args:
        logits: Model predictions (B, seq_len, vocab_size)
        labels: Ground truth labels (B, seq_len)
        combined_embeddings: Visual-text embeddings (optional)
        model_outputs: Full model outputs for additional signals
        loss_weight: Weight for loss component
        uncertainty_weight: Weight for uncertainty component
        diversity_weight: Weight for diversity component
        confidence_threshold: Threshold for considering a prediction uncertain
        global_priorities: Current priorities in the buffer (for temporal smoothing)
        
    Returns:
        scores: Improved informativeness scores for each sample in batch
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    batch_size = logits.size(0)
    
    # Create mask for valid tokens (not -100)
    mask = (labels != -100).float()
    
    # ========== COMPONENT 1: LOSS CALCULATION ==========
    # Calculate per-token loss
    loss_per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none',
        ignore_index=-100
    ).view(batch_size, -1)
    
    # Average loss per sample
    num_valid_tokens = mask.sum(dim=1)
    loss_per_sample = (loss_per_token * mask).sum(dim=1) / torch.clamp(num_valid_tokens, min=1.0)
    
    # Normalize the loss to [0,1] range
    if batch_size > 1:
        loss_min = loss_per_sample.min()
        loss_max = loss_per_sample.max()
        if loss_max > loss_min:
            norm_loss = (loss_per_sample - loss_min) / (loss_max - loss_min)
        else:
            norm_loss = torch.ones_like(loss_per_sample) * 0.5
    else:
        # For single sample batches
        norm_loss = torch.ones_like(loss_per_sample) * 0.5
    
    # ========== COMPONENT 2: UNCERTAINTY CALCULATION ==========
    # Get confidence scores for predictions
    probabilities = F.softmax(logits, dim=-1)
    predicted_indices = torch.argmax(logits, dim=-1)
    
    # Get confidence for each predicted token
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, probabilities.size(1))
    seq_indices = torch.arange(probabilities.size(1)).unsqueeze(0).expand(batch_size, -1)
    
    if predicted_indices.device != probabilities.device:
        predicted_indices = predicted_indices.to(probabilities.device)
    
    confidence = probabilities[batch_indices, seq_indices, predicted_indices]
    avg_confidence = (confidence * mask).sum(dim=1) / torch.clamp(num_valid_tokens, min=1.0)
    
    # Calculate uncertainty (inversely related to confidence)
    uncertainty_score = 1.0 - avg_confidence  # Linear mapping from confidence
    
    # ========== COMPONENT 3: DIVERSITY BONUS ==========
    # Add diversity bonus based on embedding similarity if available
    diversity_bonus = torch.zeros_like(loss_per_sample)
    
    if combined_embeddings is not None and batch_size > 1:
        # Calculate cosine similarity between embeddings
        norm_embeddings = F.normalize(combined_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.transpose(0, 1))
        
        # Average similarity to other samples (excluding self-similarity)
        similarity_matrix.fill_diagonal_(0)
        avg_similarity = similarity_matrix.sum(dim=1) / (batch_size - 1)
        
        # Diversity bonus (inversely related to similarity)
        diversity_bonus = 1.0 - avg_similarity
    
    # ========== COMBINE COMPONENTS WITH WEIGHTS ==========
    informativeness = (
        loss_weight * norm_loss + 
        uncertainty_weight * uncertainty_score + 
        diversity_weight * diversity_bonus
    )
    
    # Add temporal smoothing if global priorities are provided
    if global_priorities is not None:
        # Apply exponential moving average
        alpha = 0.7  # Smoothing factor
        informativeness = alpha * informativeness + (1 - alpha) * global_priorities
    
    # Clip and return as numpy array
    return torch.clamp(informativeness, min=0.01, max=1.0).detach().cpu().numpy()