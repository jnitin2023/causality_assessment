# src/data_preprocessing.py

import pandas as pd
import torch
from transformers import GPT2Tokenizer

# Mock data
data = {
    'description': [
        'Patient experienced severe headache and nausea after taking medication A.',
        'No side effects observed with medication B after 1 month of use.',
        'Mild rash appeared on the arms and legs after the patient took medication C for a week.',
        'Patient reported severe dizziness and loss of balance after taking medication D for three days.',
        'No adverse effects noted with medication E even after two months of continuous use.',
        'Patient experienced shortness of breath and chest pain after the first dose of medication F.',
        'No side effects observed with medication G in the first week.',
        'Patient developed severe allergic reaction and skin rash after taking medication H.',
        'Patient reported fatigue and muscle weakness after taking medication I for two weeks.',
        'No adverse effects were reported with medication J during the initial trial period.'
    ],
    'causality': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 1 indicates causality, 0 indicates no causality
}

df = pd.DataFrame(data)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# Apply preprocessing
tokenized_data = [preprocess(desc) for desc in df['description']]
input_ids = [data['input_ids'].squeeze(0) for data in tokenized_data]
attention_mask = [data['attention_mask'].squeeze(0) for data in tokenized_data]

df['input_ids'] = input_ids
df['attention_mask'] = attention_mask
df.to_pickle('processed_data_llm.pkl')
