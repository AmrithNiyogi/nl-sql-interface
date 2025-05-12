# train_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
import torch

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Load the pre-trained model (GPT-Neo as an alternative to Llama)
model_name = "EleutherAI/gpt-neo-1.3B"  # Use GPT-Neo (or you can choose GPT-J)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenizing the data (questions + answers)
def encode_data(question, answer):
    input_text = f"Question: {question} Answer: {answer}"
    return tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

# Prepare the dataset
train_data = [encode_data(row['question'], row['answer']) for _, row in data.iterrows()]

# Create a PyTorch Dataset
class SQLDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = SQLDataset(train_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')

print("Fine-tuning complete and model saved to 'fine_tuned_model'.")
