# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict

# -------------------------------
# 1️⃣ Load Dataset from CSV
# -------------------------------
df = pd.read_csv("riddles.csv")
texts = [f"Riddle: {row['Riddle']}\nAnswer: {row['Answer']}" for _, row in df.iterrows()]
dataset = Dataset.from_dict({"text": texts})

# Split into training and validation sets
dataset = dataset.train_test_split(test_size=0.2)

# -------------------------------
# 2️⃣ Load Pre-trained GPT-2 Model
# -------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# -------------------------------
# 3️⃣ Fine-tune GPT-2 at Runtime
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",  # Disable Weights & Biases
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=2,  # Adjust for speed
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()

# -------------------------------
# 4️⃣ Riddle Generation Function
# -------------------------------
def generate_riddle():
    input_text = "Generate a tricky math riddle: "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
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

    riddle = tokenizer.decode(outputs[0], skip_special_tokens=True)
    riddle = riddle.split("\n")[0]  # Extract only the relevant part
    return riddle

# -------------------------------
# 5️⃣ Streamlit UI for Deployment
# -------------------------------
st.set_page_config(page_title="Math Riddle Generator", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stTextArea textarea {
            font-size: 16px;
            background-color: #f0f0f0;
            border-radius: 8px;
        }
        .title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='title'>Math Riddle Generator 🧠</p>", unsafe_allow_html=True)
st.write("Generate unique and fun math riddles with AI!")

if st.button("Generate Riddle"):
    riddle = generate_riddle()
    st.text_area("Generated Riddle", riddle, height=150)
