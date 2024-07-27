from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CausalityModel:
    def __init__(self, model_path='./results', tokenizer_path='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def predict(self, description: str):
        inputs = self.tokenizer(description, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        return prediction, probabilities.tolist()
