import torch
import streamlit as st
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# ✅ **Step 1: Load Riddles Dataset**
@st.cache_data
def load_dataset():
    file_path = "riddles.csv"  # Path to the uploaded file
    df = pd.read_csv(file_path)  # Load CSV
    df = df.dropna()  # Remove empty rows if any
    return df

df = load_dataset()
riddles = [{"riddle": row["riddle"], "answer": row["answer"]} for _, row in df.iterrows()]

# ✅ **Step 2: Tokenizer & Model Setup**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# ✅ **Step 3: Convert Dataset for Training**
dataset = Dataset.from_dict({"text": [f"Q: {r['riddle']} A: {r['answer']}" for r in riddles]})
dataset = dataset.map(lambda x: {"input_ids": tokenizer(x["text"], truncation=True, padding="max_length", max_length=64)["input_ids"]})

# ✅ **Step 4: Fine-Tuning GPT-2**
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    no_cuda=not torch.cuda.is_available(),  # Force CPU mode if no GPU
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ✅ **Step 5: Riddle Generation Function**
def generate_riddle():
    input_text = "Q: "  # Prompt for riddle generation
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# ✅ **Step 6: Streamlit UI**
st.set_page_config(page_title="Math Riddle Generator", layout="centered")
st.title("🤖 Math Riddle Factory")

if st.button("Generate Riddle"):
    riddle = generate_riddle()
    st.write(f"🧩 **Riddle:** {riddle}")
