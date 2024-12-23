import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from app.config import MODEL_DIR, TOKENIZER_DIR, DEVICE

model = None
tokenizer = None

def load_model_and_tokenizer():
    """ Load model and tokenizer lazily """
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")
        if not os.path.exists(MODEL_DIR) or not os.path.exists(TOKENIZER_DIR):
            raise FileNotFoundError("Model or tokenizer not found in the specified directories.")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    return model, tokenizer