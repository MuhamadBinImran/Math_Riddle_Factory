import os
import pandas as pd
import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# ✅ Move set_page_config to the first line
st.set_page_config(page_title="Math Riddle Generator", page_icon="🧠", layout="centered")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset dynamically
@st.cache_data
def load_dataset():
    df = pd.read_csv("riddles.csv")  # Ensure riddles.csv is uploaded in Streamlit
    texts = [f"Riddle: {row['Riddle']}\nAnswer: {row['Answer']}" for _, row in df.iterrows()]
    dataset = Dataset.from_dict({"text": texts})
    return dataset.train_test_split(test_size=0.2)

dataset = load_dataset()

# Cached function to load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    return model, tokenizer

model, tokenizer = load_model()

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training function
def fine_tune():
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        num_train_epochs=2,  # Reduced to 2 for faster execution
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="no",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()
    st.success("✅ Model fine-tuned at runtime!")

# Function to generate a riddle
def generate_riddle():
    input_text = "Generate a tricky math riddle: "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("Math Riddle Generator 🧠")

if st.button("Fine-tune Model (Quick)"):
    fine_tune()

if st.button("Generate Riddle"):
    riddle = generate_riddle()
    st.text_area("Generated Riddle", riddle, height=100)
