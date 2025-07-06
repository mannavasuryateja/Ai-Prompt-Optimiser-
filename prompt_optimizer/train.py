from optimizer.model import load_model_and_tokenizer
from optimizer.reward import reward_function
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset
import torch

model, tokenizer = load_model_and_tokenizer()

data = [
    {"original": "Write a poetic description of the moon during a solar eclipse.",
     "compressed": "Describe the moon in a solar eclipse poetically."},
]
dataset = Dataset.from_list(data)

def tokenize(batch):
    return tokenizer(batch['original'], padding="max_length", truncation=True, max_length=64)

def tokenize_labels(batch):
    return tokenizer(batch['compressed'], padding="max_length", truncation=True, max_length=64)

tokenized_inputs = dataset.map(tokenize)
tokenized_labels = dataset.map(tokenize_labels)
tokenized_inputs["labels"] = tokenized_labels["input_ids"]

training_args = TrainingArguments(
    output_dir="optimizer_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_inputs,
    tokenizer=tokenizer,
)

trainer.train()
