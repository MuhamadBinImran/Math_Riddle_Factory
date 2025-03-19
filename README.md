# Math Riddle Generator

A deep learning model fine-tuned on GPT-2 to generate tricky and fun math riddles. This project fine-tunes a transformer-based model using a dataset of math riddles and deploys it via a Gradio web interface.

## 🚀 Features
- Fine-tunes GPT-2 to generate math riddles.
- Uses the `transformers` library for training.
- Deploys an interactive web UI with Gradio.
- Generates unique riddles at the click of a button.

## 📦 Installation
Clone the repository and install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/MuhamadBinImran/math-riddle-generator.git
cd math-riddle-generator

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset
Ensure you have a dataset of riddles in `riddles.csv` format:
```csv
Riddle,Answer
"What number becomes zero when you subtract 15 from half of it?",30
"I am a three-digit number. My tens digit is five more than my ones digit, and my hundreds digit is eight less than my tens digit. What number am I?",194
...
```

## 🔧 Training the Model
Fine-tune GPT-2 on the riddle dataset:
```sh
python train.py
```
This script:
- Loads `riddles.csv`.
- Tokenizes and fine-tunes GPT-2.
- Saves the model in `./fine_tuned_model`.

## 🎭 Running the Gradio App
Launch the Gradio web interface:
```sh
python app.py
```
After running, a link will appear in the terminal. Open it in your browser to generate riddles!

## 📜 Usage
- Click **Generate Riddle** to get a new riddle.
- The model generates a unique math riddle based on its fine-tuned knowledge.

## 📂 Project Structure
```
math-riddle-generator/
│── riddles.csv          # Dataset
│── train.py             # Training script
│── app.py               # Gradio UI
│── fine_tuned_model/    # Saved model (after training)
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

## 🛠 Dependencies
Install the required Python libraries:
```sh
pip install torch transformers datasets gradio pandas
```

## 📝 License
This project is licensed under the MIT License.

