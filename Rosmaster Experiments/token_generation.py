import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import T5Tokenizer, T5EncoderModel

def generate_vqa_tokens(vision_encoder, text_encoder, text_embedd_dim, tokenizer, json_file_path, device):
    # Projection maps T5 encoder's hidden size down to 16 dims
    text_projection = nn.Linear(text_encoder.config.hidden_size, text_embedd_dim).to(device)
    
    # If your vision encoder was trained with ImageNet normalization, uncomment and set mean/std:
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalize
    ])
    
    vision_encoder.eval()
    text_encoder.eval()
    text_projection.eval()
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        # --- Vision side ---
        image = Image.open(item['image_path']).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        with torch.no_grad():
            *_, vision_embedding = vision_encoder(image_tensor)    # [1, 24]
        
        # --- Text side ---
        q_texts, q_embs, ans_token_ids = [], [], []
        for q in item['questions']:
            question, answer = q['question'], q['answer']
            q_texts.append(question)
            
            # Tokenize question and move tensors to device
            inputs = tokenizer(
                question,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=75
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = text_encoder(**inputs)
            pooled = out.last_hidden_state.mean(dim=1)            # [1, hidden_size]
            proj   = text_projection(pooled)                      # [1, 16]
            q_embs.append(proj)
            
            # Tokenize answer
            ans_inputs = tokenizer(
                answer,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            ans_ids = ans_inputs.input_ids.to(device)            # [1, seq_len]
            ans_token_ids.append(ans_ids)
        
        # Combine vision + text embeddings
        q_embs   = torch.cat(q_embs, dim=0)                      # [N, 16]
        v_embs   = vision_embedding.repeat(len(q_embs), 1)      # [N, 24]
        combined = torch.cat([v_embs, q_embs], dim=1)           # [N, 40]
        
        results.append({
            'image_path': item['image_path'],
            'questions': q_texts,
            'combined_embeddings': combined,
            'answer_tokens': ans_token_ids
        })
    
    return results