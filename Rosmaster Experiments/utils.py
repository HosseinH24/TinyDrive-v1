import torch
from torch import nn
from torchvision import transforms
from torch.utils import data
import cv2
from PIL import Image
import json
import glob
from sklearn.model_selection import train_test_split
from ptflops import get_model_complexity_info
import tqdm
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, ReduceLROnPlateau
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer, T5ForConditionalGeneration, logging
from torch.amp import GradScaler, autocast
import random
import datetime
from prioritized_buffer import PrioritizedVQADataset, compute_informativeness_score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import random
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

logging.set_verbosity_error()

trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def get_only_images():

    all_images = []
    all_labels = []

    """
    read the .json file containing path to object images
    """
    with open('path_to/visual_qa.json') as f:
        json_data = json.load(f)


    """
    get the object images, append them all_object_images as well as all_images
    """
    all_object_images = []
    for item in json_data:
        object_image_path = item["image_path"]
        image_label = item["class_id"]
        object_img = Image.open(object_image_path).convert("RGB")
        object_img = trans(object_img)
        all_object_images.append(object_img)
        all_images.append(object_img)
        all_labels.append(image_label)


    """
    get no-object images, append them to all_no_object_images as well as all_images
    """
    
    
    all_no_object_images = []
    """
    for item in glob.glob('/Users/hosseinhassani/Desktop/VLM/data with speed/final_data_v02_640x480/no_object_images/*.jpg'):
        no_object_img = Image.open(item).convert("RGB")
        no_object_img = trans(no_object_img)
        all_no_object_images.append(no_object_img)
        all_images.append(no_object_img)
        all_labels.append(10)
    """

    print(f"Object Images: {len(all_object_images)}, No-Object Images: {len(all_no_object_images)}, Total: {len(all_images)}")

    return all_images, all_labels


def create_train_val_loaders(all_images, all_labels, BATCH):

    train_x, val_x, train_y, val_y = train_test_split(all_images, all_labels, test_size=0.1, random_state=42)
    train_x = torch.stack(train_x)
    val_x = torch.stack(val_x)
    train_y = torch.LongTensor(train_y)
    val_y = torch.LongTensor(val_y)

    train_dataset = data.TensorDataset(*(train_x, train_y))
    val_dataset = data.TensorDataset(*(val_x, val_y))

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    return train_loader, val_loader


def get_accuracy(preds, org):
    pred_class = torch.argmax(preds, dim=1)
    correct = (pred_class == org).sum().item()
    total = org.size(0)
    return correct / total

def count_parameters(model):

    return sum([p.numel() for p in model.parameters() if p.requires_grad==True])

def count_flops(model, input_res):
    with torch.cuda.device(0):  # or `with torch.device('cpu')` if no GPU
        macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
    return macs * 2  # 1 MAC = 2 FLOPs


def get_an_image_tensor(path):
    image = Image.open(path).convert("RGB")

    return trans(image)


def compute_loss(loss_function, logits, attn_high, attn_mid, attn_low, labels, lambda_attn):

    ce_loss = loss_function(logits, labels)

    attn_high_norm = torch.abs(attn_high).mean()
    attn_mid_norm = torch.abs(attn_mid).mean()
    attn_low_norm = torch.abs(attn_low).mean()

    attn_loss = attn_high_norm #+ attn_mid_norm + attn_low_norm

    total_loss = ce_loss + lambda_attn * attn_loss

    return total_loss, ce_loss


def train_vision_encoder(model, epochs, optimizer, device, train_loader, val_loader):

    path_to_model = 'path_to/vision encoder models'
    best_model = os.path.join(path_to_model, "best_model.pth")
    final_model = os.path.join(path_to_model, "final_model.pth")
    f2_model = os.path.join(path_to_model, "f2_model.pth")
    best_val_acc = 0.0

    classification_loss_function = nn.CrossEntropyLoss()
    #scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    for epoch in range(epochs):

        epoch_loss = []
        epoch_train_acc = []
        epoch_val_acc = []
        epoch_overall_loss = []

        model.train()

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            class_logits, attn_high, attn_mid, attn_low, vision_embeds = model(X)
            #classification_loss = classification_loss_function(class_logits, y)
            overall_loss, classification_loss = compute_loss(classification_loss_function, class_logits, attn_high, attn_mid, attn_low, y, lambda_attn=0.0)

            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()

            train_acc = get_accuracy(class_logits, y)

            epoch_loss.append(classification_loss.detach().cpu())
            epoch_train_acc.append(train_acc)
            epoch_overall_loss.append(overall_loss.detach().cpu())

        model.eval()

        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            class_logits, attn_high, attn_mid, attn_low, vision_embeds = model(X)
            val_loss = classification_loss_function(class_logits, y)
            val_acc = get_accuracy(class_logits, y)
            epoch_val_acc.append(val_acc)

        scheduler.step()

        print("-------------------------------------------------")
        print(f"epoch: {epoch+1}/{epochs}")
        print(f"loss: {np.array(epoch_loss).mean():.4f}, train acc: {np.array(epoch_train_acc).mean():.4f}, val acc: {np.array(epoch_val_acc).mean():.4f}")
        #print(f"Overall       ---> loss: {np.array(epoch_overall_loss).mean():.4f}")

        if (np.array(epoch_val_acc).mean() + np.array(epoch_train_acc).mean()) / 2 > best_val_acc:
            torch.save(model.state_dict(), best_model)
            best_val_acc = (np.array(epoch_val_acc).mean() + np.array(epoch_train_acc).mean()) / 2
            print(f"Best Saved: {(np.array(epoch_val_acc).mean() + np.array(epoch_train_acc).mean()) / 2:.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), final_model)
            #print("final model saved.")

    torch.save(model.state_dict(), f2_model)
    print("f2 model saved.")

def compute_token_importance(input_ids, embeddings, tokenizer, combined_embeddings=None):
    batch_size, seq_len, embed_dim = embeddings.shape
    device = embeddings.device
    
    importance_scores = torch.zeros((batch_size, seq_len), device=device)
    
    token_magnitudes = torch.norm(embeddings, dim=2)
    importance_scores += token_magnitudes
    
    pos_weights = torch.ones((seq_len), device=device)
    if seq_len > 5:
        mid_start = seq_len // 4
        mid_end = 3 * seq_len // 4
        pos_weights[mid_start:mid_end] = 1.2
    importance_scores *= pos_weights.unsqueeze(0)
    
    is_special = torch.zeros((batch_size, seq_len), device=device)
    
    pad_mask = (input_ids != tokenizer.pad_token_id).float()
    
    importance_scores = importance_scores * pad_mask
    
    row_max, _ = importance_scores.max(dim=1, keepdim=True)
    row_min, _ = importance_scores.min(dim=1, keepdim=True)
    denom = row_max - row_min
    denom[denom == 0] = 1.0
    importance_scores = (importance_scores - row_min) / denom
    
    return importance_scores

def apply_topk_masking(embeddings, importance_scores, k=64, vision_embed_len=24):
    batch_size, seq_len, embed_dim = embeddings.shape
    device = embeddings.device
    
    masked_embeddings = embeddings.clone()
    
    for i in range(batch_size):
        text_scores = importance_scores[i, :-vision_embed_len] if vision_embed_len > 0 else importance_scores[i]
        
        if text_scores.shape[0] > k:
            _, top_indices = torch.topk(text_scores, k)
            
            mask = torch.zeros(seq_len - vision_embed_len, device=device)
            mask[top_indices] = 1.0
            
            masked_embeddings[i, :-vision_embed_len] *= mask.unsqueeze(1)
    
    return masked_embeddings

def fine_tune_language_model(
    train_tokens,
    val_tokens,
    test_tokens,
    language_model,
    vision_encoder,
    tokenizer,
    device,
    epochs=50,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-2,
    projection_input=40,
    alpha=0.6,
    beta_start=0.4,
    buffer_capacity=5_000,
    top_k_text=64,
    vision_embed_count=24,
):
    tokenizer.extra_ids = 0
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    language_model.resize_token_embeddings(len(tokenizer))

    projection = nn.Sequential(
        nn.Linear(projection_input, 512),
        nn.ReLU(),
        nn.Linear(512, language_model.config.d_model),
    ).to(device)

    print("Creating prioritized datasets...")
    train_dataset = PrioritizedVQADataset(
        train_tokens, 
        tokenizer, 
        device, 
        capacity=buffer_capacity, 
        alpha=alpha, 
        beta=beta_start
    )
    val_dataset = PrioritizedVQADataset(
        val_tokens, 
        tokenizer, 
        device, 
        capacity=min(buffer_capacity, len(val_tokens)*5), 
        alpha=alpha, 
        beta=beta_start
    )
    test_dataset = PrioritizedVQADataset(
        test_tokens, 
        tokenizer, 
        device, 
        capacity=min(buffer_capacity, len(test_tokens)*5), 
        alpha=alpha, 
        beta=beta_start
    )

    def collate_fn(batch):
        qids = [b['question_input_ids'] for b in batch]
        labs = [b['labels'] for b in batch]
        embs = torch.stack([b['combined_embedding'] for b in batch], dim=0)

        max_q = max(q.shape[0] for q in qids)
        input_ids = torch.stack([
            nn.functional.pad(q, (0, max_q - q.shape[0]), value=tokenizer.pad_token_id)
            for q in qids
        ], dim=0)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        max_l = max(l.shape[0] for l in labs)
        labels = torch.stack([
            nn.functional.pad(l, (0, max_l - l.shape[0]), value=-100)
            for l in labs
        ], dim=0)

        raw_data = [b.get('raw_data', None) for b in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'combined_embeddings': embs,
            'labels': labels,
            'raw_data': raw_data
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    optimizer = torch.optim.AdamW(
        [
            {'params': language_model.parameters(), 'lr': learning_rate},
            {'params': projection.parameters(), 'lr': learning_rate},
            {'params': vision_encoder.parameters(), 'lr': learning_rate / 10},
        ],
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    language_model.train()
    vision_encoder.train()

    best_val_loss = float('inf')
    save_dir = os.path.join(os.getcwd(), "language_model")
    os.makedirs(save_dir, exist_ok=True)
    
    priority_updates = 0
    mean_priorities = []

    for epoch in range(1, epochs + 1):
        total_train_loss = 0.0
        language_model.train()
        vision_encoder.train()
        batch_indices = []
        batch_priorities = []
        
        train_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            combined_embeddings = batch['combined_embeddings'].to(device)

            q_embeds = language_model.get_input_embeddings()(input_ids)
            
            img_proj = projection(combined_embeddings).unsqueeze(1)
            
            inputs_embeds = torch.cat([q_embeds, img_proj], dim=1)
            
            importance_scores = compute_token_importance(
                input_ids, 
                q_embeds, 
                tokenizer, 
                combined_embeddings
            )
            
            masked_inputs_embeds = apply_topk_masking(
                inputs_embeds, 
                torch.cat([importance_scores, torch.ones(importance_scores.size(0), 1, device=device)], dim=1),
                k=top_k_text, 
                vision_embed_len=1
            )

            bsz = attention_mask.size(0)
            attn_mask = torch.cat([
                attention_mask,
                torch.ones((bsz, 1), dtype=torch.long, device=device)
            ], dim=1)

            outputs = language_model(
                inputs_embeds=masked_inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,
                output_hidden_states=True
            )
            loss = outputs.loss

            if torch.isnan(loss):
                print("⚠️ Skipping step due to NaN loss")
                continue

            with torch.no_grad():
                informativeness = compute_informativeness_score(
                    outputs.logits, 
                    labels,
                    combined_embeddings
                )
                
                indices = list(range(batch_idx * batch_size, 
                                    min((batch_idx + 1) * batch_size, len(train_dataset))))
                batch_indices.extend(indices[:len(informativeness)])
                batch_priorities.extend(informativeness.tolist())

            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(language_model.parameters()) + list(projection.parameters()) + list(vision_encoder.parameters()), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
            if batch_idx % 5 == 0 and batch_indices:
                train_dataset.update_priorities(batch_indices, batch_priorities)
                priority_updates += 1
                mean_priorities.append(np.mean(batch_priorities))
                batch_indices = []
                batch_priorities = []
        
        if batch_indices:
            train_dataset.update_priorities(batch_indices, batch_priorities)
            priority_updates += 1
            mean_priorities.append(np.mean(batch_priorities))
        
        train_time = time.time() - train_start
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)

        language_model.eval()
        vision_encoder.eval()
        total_val_loss = 0.0
        val_indices = []
        val_priorities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                combined_embeddings = batch['combined_embeddings'].to(device)

                q_embeds = language_model.get_input_embeddings()(input_ids)
                img_proj = projection(combined_embeddings).unsqueeze(1)
                inputs_embeds = torch.cat([q_embeds, img_proj], dim=1)
                
                importance_scores = compute_token_importance(
                    input_ids, 
                    q_embeds, 
                    tokenizer, 
                    combined_embeddings
                )
                
                masked_inputs_embeds = apply_topk_masking(
                    inputs_embeds, 
                    torch.cat([importance_scores, torch.ones(importance_scores.size(0), 1, device=device)], dim=1),
                    k=top_k_text, 
                    vision_embed_len=1
                )

                bsz = attention_mask.size(0)
                attn_mask = torch.cat([
                    attention_mask,
                    torch.ones((bsz, 1), dtype=torch.long, device=device)
                ], dim=1)

                outputs = language_model(
                    inputs_embeds=masked_inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                    output_hidden_states=True
                )
                loss = outputs.loss
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                
                informativeness = compute_informativeness_score(
                    outputs.logits, 
                    labels,
                    combined_embeddings
                )
                
                indices = list(range(batch_idx * batch_size, 
                                    min((batch_idx + 1) * batch_size, len(val_dataset))))
                val_indices.extend(indices[:len(informativeness)])
                val_priorities.extend(informativeness.tolist())

        if val_indices:
            val_dataset.update_priorities(val_indices, val_priorities)

        avg_val_loss = total_val_loss / len(val_loader)
        
        if mean_priorities:
            avg_priority = np.mean(mean_priorities)
            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                 f"Avg Priority: {avg_priority:.4f}, Time: {train_time:.2f}s")
            mean_priorities = []
        else:
            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                 f"Time: {train_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            language_model.save_pretrained(save_dir, safe_serialization=False, save_format='pt')
            tokenizer.save_pretrained(save_dir)
            torch.save(projection.state_dict(), os.path.join(save_dir, "projection.pt"))
            vision_encoder_cpu = vision_encoder.cpu()
            torch.save(vision_encoder_cpu.state_dict(), os.path.join(save_dir, "vision_encoder.pt"))

    print("\n=== Evaluating on Test Set ===")
    metrics = evaluate_model(
        language_model,
        vision_encoder,
        projection,
        tokenizer,
        test_loader,
        device,
        top_k_text=top_k_text,
        sample_count=10
    )
    
    return language_model, projection, vision_encoder

def evaluate_model(
    language_model,
    vision_encoder,
    projection,
    tokenizer,
    test_loader,
    device,
    top_k_text=64,
    sample_count=10
):
    language_model.eval()
    vision_encoder.eval()
    
    all_predictions = []
    all_references = []
    all_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            combined_embeddings = batch['combined_embeddings'].to(device)
            raw_data = batch['raw_data']
            
            q_embeds = language_model.get_input_embeddings()(input_ids)
            img_proj = projection(combined_embeddings).unsqueeze(1)
            inputs_embeds = torch.cat([q_embeds, img_proj], dim=1)
            
            importance_scores = compute_token_importance(
                input_ids, 
                q_embeds, 
                tokenizer, 
                combined_embeddings
            )
            
            masked_inputs_embeds = apply_topk_masking(
                inputs_embeds, 
                torch.cat([importance_scores, torch.ones(importance_scores.size(0), 1, device=device)], dim=1),
                k=top_k_text, 
                vision_embed_len=1
            )

            bsz = attention_mask.size(0)
            attn_mask = torch.cat([
                attention_mask,
                torch.ones((bsz, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            outputs = language_model.generate(
                inputs_embeds=masked_inputs_embeds,
                attention_mask=attn_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            references = []
            for label in labels:
                filtered_label = label[label != -100]
                if len(filtered_label) > 0:
                    reference = tokenizer.decode(filtered_label, skip_special_tokens=True)
                    references.append(reference)
                else:
                    references.append("")
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            for i in range(len(predictions)):
                if raw_data[i] is not None:
                    sample = {
                        'image_path': raw_data[i].get('image_path', "Unknown"),
                        'question': raw_data[i].get('question', "Unknown"),
                        'generated_answer': predictions[i],
                        'reference_answer': references[i]
                    }
                    all_samples.append(sample)
    
    smoother = SmoothingFunction().method4
    bleu_scores = []
    for ref, pred in zip(all_references, all_predictions):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        if len(pred_tokens) == 0:
            bleu_scores.append(0.0)
        else:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
            bleu_scores.append(score)
    bleu4 = np.mean(bleu_scores)
    
    tokenized_refs = [ref.split() for ref in all_references]
    tokenized_preds = [pred.split() for pred in all_predictions]
    
    meteor_scores = []
    for ref, pred in zip(tokenized_refs, tokenized_preds):
        score = meteor_score([ref], pred)
        meteor_scores.append(score)
    meteor = np.mean(meteor_scores)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for ref, pred in zip(all_references, all_predictions):
        if len(pred.strip()) == 0 or len(ref.strip()) == 0:
            rouge_scores.append(0.0)
            continue
        
        try:
            scores = scorer.score(ref, pred)
            rouge_scores.append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            rouge_scores.append(0.0)
    
    rouge_l = np.mean(rouge_scores)
    
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(
        {i: [all_references[i]] for i in range(len(all_references))},
        {i: [all_predictions[i]] for i in range(len(all_predictions))}
    )
    
    print(f"\n=== Test Metrics ===")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"CIDEr: {cider_score:.4f}")
    
    # Print 10 random samples from all_samples
    print("\n=== Sample Responses from Test Dataset ===")
    if len(all_samples) > 0:
        # Randomly select 10 samples (or all if fewer than 10)
        selected_samples = random.sample(all_samples, min(sample_count, len(all_samples)))
        for idx, sample in enumerate(selected_samples, 1):
            print(f"\nSample {idx}:")
            print(f"Question: {sample['question']}")
            print(f"Image Path: {sample['image_path']}")
            print(f"Predicted Response: {sample['generated_answer']}")
            print(f"Ground Truth: {sample['reference_answer']}")
    else:
        print("No samples available to display.")
    
    metrics = {
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l,
        'cider': cider_score
    }
    
    return metrics

def process_tokens_for_training(token_results, vision_encoder):
    processed_tokens = []
    
    for item in token_results:
        processed_item = {
            'image_path': item['image_path'],
            'questions': item['questions'],
            'combined_embeddings': item['combined_embeddings'],
            'answer_tokens': item['answer_tokens']
        }
        processed_tokens.append(processed_item)
    
    return processed_tokens

def extract_vision_embeddings(vision_encoder, image_tensor, device):
    vision_encoder.eval()
    with torch.no_grad():
        class_logits, attn_high, attn_med, attn_low, vision_embeddings = vision_encoder(image_tensor.to(device))
    return vision_embeddings
