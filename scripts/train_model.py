# train_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm  # Import tqdm

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Format the input texts as plain text (for causal LM training)
def format_example(row):
    return {
        "text": f"Question: {row['question']} Answer: {row['answer']}"
    }

formatted_data = data.apply(format_example, axis=1)
dataset = Dataset.from_pandas(pd.DataFrame(formatted_data.tolist()))

# Load tokenizer and model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for GPT-Neo

# Tokenize the dataset with tqdm to show progress
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# Wrap dataset processing with tqdm
tokenized_dataset = dataset.map(lambda x: tokenize_function(x), batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

def tokenize_with_progress(example_batch):
    return {key: tqdm(value, desc=f"Tokenizing {key}") for key, value in tokenize_function(example_batch).items()}

tokenized_dataset = dataset.map(
    lambda x: tokenize_with_progress(x),
    batched=True
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-Neo is a causal LM, not masked
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model with tqdm for training loop
print("Starting training...")
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("Fine-tuning complete and model saved to 'fine_tuned_model'.")
