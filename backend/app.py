# app.py
# Updated to use your fine-tuned LoRA model.

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import pdfplumber
import os

# --- Configuration ---
# Base model used for fine-tuning
BASE_MODEL_NAME = "google/flan-t5-base" 
# Path to your fine-tuned LoRA adapter
# PEFT_MODEL_PATH = "flan-legal-lora/checkpoint-5454"
PEFT_MODEL_PATH = "../legal-flan-t5-finetune/flan-legal-lora/checkpoint-5454"

# --- Initialize Flask App ---
app = Flask(__name__)
# CORS is needed to allow your frontend to make requests to this backend
CORS(app)

# --- Model Loading ---

print("Loading base model and tokenizer...")
# Use BitsAndBytesConfig for 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)

print("Applying the fine-tuned LoRA adapter...")
# Apply the LoRA adapter to the base model
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
# Set the model to evaluation mode for faster inference
model.eval()

print("Model loaded successfully. Flask app is ready.")


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """
    This endpoint now uses your fine-tuned model for inference.
    """
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'response': "Error: Empty prompt received."}), 400

    # Format the input exactly as it was during training for best results
    input_text = f"Legal Q: {prompt}"
    print(f"Received prompt: '{prompt}' -> Formatted as: '{input_text}'")

    try:
        # Tokenize the formatted input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

        # Generate the response from the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                max_length=2048,
                num_beams=4, # Use beam search for higher quality output
                early_stopping=True
            )
        
        # Decode the generated tokens back into text
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated answer: {answer}")
        
        return jsonify({'response': answer.strip()})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'response': f"Model Error: {str(e)}"}), 500


@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    This endpoint for PDF processing remains unchanged.
    """
    file = request.files.get('pdf')
    if not file:
        return jsonify({'text': ''})

    try:
        text = ''
        with pdfplumber.open(file) as pdf:
            # Limiting to the first 3 pages to keep processing fast
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'

        return jsonify({'text': text.strip()})
    except Exception as e:
        return jsonify({'text': f"PDF processing error: {str(e)}"}), 500


if __name__ == '__main__':
    # Running with debug=True is fine for development
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
