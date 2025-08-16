import torch
from torch import nn
from utils import count_parameters, count_flops, train_vision_encoder, get_only_images, create_train_val_loaders
from utils import fine_tune_language_model
from vision_encoder2 import MultiScaleVisionEncoder
import numpy as np
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5EncoderModel, T5Config
from token_generation import generate_vqa_tokens
import os

def main():

	language_model_name = "google/t5-efficient-tiny"

	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	print(f"Device: {device}")
	
	tokenizer = T5Tokenizer.from_pretrained(language_model_name)
	text_encoder = T5EncoderModel.from_pretrained(language_model_name).to(device)

	VisionEPOCHS = 30
	Vision_BATCH_SIZE = 8
	VisionLR = 1e-3
	WEIGHT_DECAY = 1e-2
	NUM_CLASS = 10

	LanguageEPOCHS = 15
	LanguageBATCH_SIZE = 4
	LanguageLR = 1e-3
	L_WEIGHT_DECAY = 1e-2

	NUM_STEM_CHANNLES = 8
	NUM_HIGH_RES_CHANNELS = 16
	NUM_MID_RES_CHANNELS = 32
	NUM_LOW_RES_CHANNELS = 64
	
	img_embedd_dim = 3 * 8
	text_embedd_dim = text_encoder.config.hidden_size
	projection_input = img_embedd_dim+text_embedd_dim

	PATH_VISUAL_QA_TRAIN = 'path_to_train_qa.json'
	PATH_VISUAL_QA_VAL = 'path_to_val_qa.json'
	PATH_VISUAL_QA_TEST = 'path_to_test_qa.json'

	
	vision_encoder = MultiScaleVisionEncoder(
		stem_channels=NUM_STEM_CHANNLES, 
		branch_channels=[NUM_HIGH_RES_CHANNELS, NUM_MID_RES_CHANNELS, NUM_LOW_RES_CHANNELS], 
		num_classes=NUM_CLASS).to(device)
	ps = count_parameters(vision_encoder)
	print(f"vision encoder params: {ps}")

	print("----------------------------------------------------------------")
	print("Generating tokens...")
	tokenizer = T5Tokenizer.from_pretrained(language_model_name)
	text_encoder = T5EncoderModel.from_pretrained(language_model_name).to(device)
	train_tokens = generate_vqa_tokens(vision_encoder, text_encoder, text_embedd_dim, tokenizer, PATH_VISUAL_QA_TRAIN, device)
	val_tokens = generate_vqa_tokens(vision_encoder, text_encoder, text_embedd_dim, tokenizer, PATH_VISUAL_QA_VAL, device)
	test_tokens = generate_vqa_tokens(vision_encoder, text_encoder, text_embedd_dim, tokenizer, PATH_VISUAL_QA_TEST, device)
	print(f"{len(train_tokens)} train tokens generated...")
	print(f"{len(val_tokens)} validation tokens generated...")
	print(f"{len(test_tokens)} test tokens generated...")

	config = T5Config.from_pretrained(language_model_name)
	config.dropout_rate=0.2
	config.attention_dropout_rate=0.2
	seq2seq_model = T5ForConditionalGeneration.from_pretrained(language_model_name).to(device)
	print("----------------------------------------------------------------")
	print("Fine tuning the language model...")
	print(f"language model params: {count_parameters(seq2seq_model)}")

	for param in vision_encoder.FC.parameters():
		param.requires_grad = False

	
	metrics = fine_tune_language_model(
	    train_tokens=train_tokens,
	    val_tokens=val_tokens,
	    test_tokens=test_tokens,
	    language_model=seq2seq_model,
	    vision_encoder=vision_encoder,
	    tokenizer=tokenizer,
	    device=device,
	    epochs=LanguageEPOCHS,
	    batch_size=LanguageBATCH_SIZE,
	    learning_rate=LanguageLR,
	    weight_decay=L_WEIGHT_DECAY,
	    projection_input=projection_input
	)
	

	path_to_model = 'path_to_language_model'
	os.makedirs(path_to_model, exist_ok=True)
	f0_model = os.path.join(path_to_model, "f0_model.pt")
	
	# Save the model while keeping it on the device
	torch.save(vision_encoder.state_dict(), f0_model)
	

	print("----------------------------------------------------------------")
	print("Fine tuning the vision encoder...")
	all_images, all_labels = get_only_images()
	train_loader, val_loader = create_train_val_loaders(all_images, all_labels, Vision_BATCH_SIZE)

	for param in vision_encoder.parameters():
		param.requires_grad = False

	for param in vision_encoder.FC.parameters():
		param.requires_grad = True

	vision_encoder = vision_encoder.to(device)
	
	vision_encoder_optimizer = torch.optim.AdamW(
		filter(lambda p: p.requires_grad, vision_encoder.parameters()), 
		lr=VisionLR, 
		weight_decay=WEIGHT_DECAY
	)

	train_vision_encoder(vision_encoder, VisionEPOCHS, vision_encoder_optimizer, device, train_loader, val_loader)
	
	f2_model = os.path.join(path_to_model, "f2_model.pth")
	torch.save(vision_encoder.state_dict(), f2_model)
	print(f"Trained vision encoder saved to {f2_model}")

if __name__ == "__main__":
	
	seed = 42
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)

	main()
