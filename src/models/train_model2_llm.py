# src/models/train_model.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load preprocessed data
df = pd.read_pickle('processed_data_llm.pkl')

# Model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Convert data to suitable format
class AdverseEventDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

encodings = {
    'input_ids': torch.stack(df['input_ids'].tolist()),
    'attention_mask': torch.stack(df['attention_mask'].tolist())
}

labels = df['causality'].tolist()
dataset = AdverseEventDataset(encodings, labels)

training_args = TrainingArguments(
    output_dir='./results2_llm',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define compute_metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics,  # Optional
)

trainer.train()

model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')
