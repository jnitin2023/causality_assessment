# preprocess_data.py
import pandas as pd
from transformers import AutoTokenizer

# Load data
data = pd.read_csv('adverse_events.csv')

# Data cleaning
data['description'] = data['description'].str.lower().str.replace('[^\w\s]', '', regex=True)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['description'], padding='max_length', truncation=True, max_length=128)

tokenized_data = data['description'].apply(lambda x: tokenize_function({'description': x}))

# Split tokenized data into separate columns
data['input_ids'] = tokenized_data.apply(lambda x: x['input_ids'])
data['attention_mask'] = tokenized_data.apply(lambda x: x['attention_mask'])

# Save the processed data
data.to_pickle('processed_data.pkl')
