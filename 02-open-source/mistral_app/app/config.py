import os
import torch

# configuration variables
MODEL_DIR = os.getenv("MODEL_DIR", "./models/mistral-7b-model")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "./models/mistral-7b-tokenizer")
DEVICE = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" and torch.cuda.is_available() else "cpu"