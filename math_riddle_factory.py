import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
import gradio as gr

# Load dataset
df = pd.read_csv("riddles.csv")
texts = [f"Riddle: {row['Riddle']}\nAnswer: {row['Answer']}" for _, row in df.iterrows()]
dataset = Dataset.from_dict({"text": texts})

# Split into training and validation sets
dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",  # Disable Weights & Biases properly
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
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

# Save model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Generate riddles with improved prompting
def generate_riddle():
    input_text = "Generate a tricky math riddle: "
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI with enhanced layout
def gradio_interface():
    riddle = generate_riddle()
    return riddle

demo = gr.Blocks()
with demo:
    gr.Markdown("## Math Riddle Generator 🧠")
    gr.Markdown("Generate unique and fun math riddles!")
    with gr.Row():
        generate_button = gr.Button("Generate Riddle")
        riddle_output = gr.Textbox(label="Generated Riddle")
    generate_button.click(fn=generate_riddle, inputs=[], outputs=riddle_output)

demo.launch(share=True)
