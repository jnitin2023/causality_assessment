import pandas as pd
from fastapi import FastAPI
from src.schemas.causality_schema import CausalityRequest
from src.models.causality_model import CausalityModel
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.routes.routes import api_router
import torch

app = FastAPI()

# Load the model
causality_model = CausalityModel()

@app.post('/assess_causality')
async def assess_causality(request: CausalityRequest):
    description = request.description
    prediction, probabilities = causality_model.predict(description)
    
    return {
        'prediction': int(prediction),
        'probabilities': probabilities
    }

# Load the fine-tuned model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('fine_tuned_model')
# tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_model')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

df = pd.read_csv('adverse_events.csv')

class AssessmentRequest(BaseModel):
    description: str

class AssessmentResponse(BaseModel):
    prediction: int
    probabilities: list

@app.post("/assess_causality_gpt", response_model=AssessmentResponse)
def assess_causality_gpt(request: AssessmentRequest):
    # Tokenize input
    inputs = tokenizer(request.description, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # For demonstration, mock the probabilities and prediction
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).tolist()
    prediction = int(torch.argmax(logits, dim=-1).item())
    
    response = AssessmentResponse(prediction=prediction, probabilities=probabilities)
    return response

@app.post('/assess_causality_llm')
async def assess_causality_llm(request: CausalityRequest):
    inputs = tokenizer(request.description, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
        prediction = torch.argmax(logits, dim=-1).item()
    
    return {
        'prediction': prediction,
        'probabilities': probabilities
    }

app.include_router(api_router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
