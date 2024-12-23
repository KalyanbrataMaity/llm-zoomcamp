from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.model_handler import load_model_and_tokenizer
import torch

router = APIRouter(prefix="/api", tags=["Text Generation"])

# Request model
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@router.post("/generate")
async def generate_text(request: TextGenerationRequest):
    """ Generate text using the Mistral model """
    try:
        tokenizer, model = load_model_and_tokenizer()
        if len(request.prompt) > 512:
            raise HTTPException(status_code=400, detail="Input text too long. Max length: 512 ccharacters.")
        
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs["input_ids"], max_length=request.max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))