import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import random
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F

class PrioritizedReplayBuffer:

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_annealing: float = 0.001):

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        
        self.epsilon = 1e-5
        
    def add(self, sample, priority=None):

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

        if self.size < batch_size:
            indices = np.random.choice(self.size, size=batch_size, replace=True)
        else:
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize to max weight
        weights = torch.FloatTensor(weights)
        
        samples = [self.buffer[idx] for idx in indices]
        
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
            
    def is_full(self):
        return self.size >= self.capacity


class PrioritizedVQADataset(Dataset):

    def __init__(self, token_entries, tokenizer, device, capacity=10000, alpha=0.6, beta=0.4):
        self.tokenizer = tokenizer
        self.device = device
        self.buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha, beta=beta)
        
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
        return self.buffer.buffer[idx]
    
    def update_priorities(self, indices, priorities):
        self.buffer.update_priorities(indices, priorities)


class PrioritizedSampler(Sampler):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
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
    
    batch_size = logits.size(0)    
    mask = (labels != -100).float()
    
    loss_per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction='none',
        ignore_index=-100
    ).view(batch_size, -1)
    
    num_valid_tokens = mask.sum(dim=1)
    loss_per_sample = (loss_per_token * mask).sum(dim=1) / torch.clamp(num_valid_tokens, min=1.0)
    
    if batch_size > 1:
        loss_min = loss_per_sample.min()
        loss_max = loss_per_sample.max()
        if loss_max > loss_min:
            norm_loss = (loss_per_sample - loss_min) / (loss_max - loss_min)
        else:
            norm_loss = torch.ones_like(loss_per_sample) * 0.5
    else:
        norm_loss = torch.ones_like(loss_per_sample) * 0.5
    

    probabilities = F.softmax(logits, dim=-1)
    predicted_indices = torch.argmax(logits, dim=-1)
    
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, probabilities.size(1))
    seq_indices = torch.arange(probabilities.size(1)).unsqueeze(0).expand(batch_size, -1)
    
    if predicted_indices.device != probabilities.device:
        predicted_indices = predicted_indices.to(probabilities.device)
    
    confidence = probabilities[batch_indices, seq_indices, predicted_indices]
    avg_confidence = (confidence * mask).sum(dim=1) / torch.clamp(num_valid_tokens, min=1.0)
    
    uncertainty_score = 1.0 - avg_confidence  # Linear mapping from confidence
    
    diversity_bonus = torch.zeros_like(loss_per_sample)
    
    if combined_embeddings is not None and batch_size > 1:
        norm_embeddings = F.normalize(combined_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.transpose(0, 1))
        
        similarity_matrix.fill_diagonal_(0)
        avg_similarity = similarity_matrix.sum(dim=1) / (batch_size - 1)
        
        diversity_bonus = 1.0 - avg_similarity
    
    informativeness = (
        loss_weight * norm_loss + 
        uncertainty_weight * uncertainty_score + 
        diversity_weight * diversity_bonus
    )
    
    if global_priorities is not None:
        alpha = 0.7
        informativeness = alpha * informativeness + (1 - alpha) * global_priorities
    
    return torch.clamp(informativeness, min=0.01, max=1.0).detach().cpu().numpy()
