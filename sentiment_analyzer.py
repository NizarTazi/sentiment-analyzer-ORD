from flask import Flask, request, jsonify, render_template
import joblib
import torch
import numpy as np
import string
from transformers import AutoTokenizer, OpenAIGPTModel
import os
import regex as re

app = Flask(__name__)

# Load the random forest model
lr_model = joblib.load('lr_GPT_sentA_model.pkl')

# Load the pre-trained OpenAI GPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-community/openai-gpt")


# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Preprocessing text functions
def gpt_tokenizer(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    return inputs

def get_gpt_embeddings(text, model, tokenizer):
    inputs = gpt_tokenizer(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move outputs to the CPU before converting to numpy

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Removing everything that is between <>
    #text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.lower()
    return text


@app.route('/')
def index():
    return render_template('index_.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'description' in request.form:
        description = request.form['description']

        # Preprocess the text
        preprocessed_text = preprocess_text(description)

        # Generate embeddings
        text_embeddings = get_gpt_embeddings(description, model, tokenizer)

        # Make predictions with the random forest model
        lr_prediction = lr_model.predict([text_embeddings])

        result = 'Positive' if lr_prediction[0] == 1 else 'Negative'

        return render_template('index_.html', prediction_text=f'Prediction: {result}')
    else:
        return render_template('index_.html', prediction_text='No prediction provided')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
