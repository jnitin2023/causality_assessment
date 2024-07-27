import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Load preprocessed data
df = pd.read_pickle('processed_data_llm.pkl')

# Model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

# Tokenize and pad data
max_length = 512

def tokenize_function(examples):
    return tokenizer(examples['description'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

# Convert data to suitable format
class AdverseEventDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenized_data = tokenize_function({'description': df['description'].tolist()})

encodings = {
    'input_ids': tokenized_data['input_ids'].squeeze(),
    'attention_mask': tokenized_data['attention_mask'].squeeze()
}

labels = df['causality'].tolist()
dataset = AdverseEventDataset(encodings, labels)

training_args = TrainingArguments(
    output_dir='./results_llm',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # You might want to split this into train and eval
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')