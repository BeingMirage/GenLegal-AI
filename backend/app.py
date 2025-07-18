# main.py
# This script creates a Flask web server to interact with a Hugging Face model.
# It provides two API endpoints:
# 1. /api/prompt: To ask legal questions and get answers from the AI model.
# 2. /api/upload-pdf: To extract text from a PDF file.

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
# NEW: Import BitsAndBytesConfig for quantization
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pdfplumber
import os

# --- Configuration ---
# Define the name of the Hugging Face model to be used.
MODEL_NAME = "varma007ut/Indian_Legal_Assitant"

# --- Initialize Flask App ---
# Create a new Flask web application.
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow requests from web frontends.
CORS(app)

# --- Device Setup ---
# Check if a CUDA-enabled GPU is available for faster processing, otherwise use the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Quantization Configuration ---
# NEW: Configure the model to be loaded in 4-bit precision.
# This significantly reduces the model's memory footprint.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# --- Load Model & Tokenizer ---
# This section handles the loading of the pre-trained model and its tokenizer from Hugging Face.
print(f"Loading model: {MODEL_NAME}...")
try:
    # Load the tokenizer associated with the model. The tokenizer converts text into a format the model understands.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the main model with the quantization config.
    # `trust_remote_code=True` is required for some custom models.
    # `device_map='auto'` helps accelerate distribute the model efficiently.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        quantization_config=quantization_config, # <-- APPLYING QUANTIZATION
        device_map="auto" # Use accelerate to map model to devices
    )
    
    # Set the model to evaluation mode (disables training-specific layers like dropout).
    model.eval()
    
    # Set the padding token to be the same as the end-of-sentence token. This is a common practice.
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded successfully in 4-bit.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model fails to load, as the application cannot function.
    exit()

# --- Alpaca-style Prompt Format ---
def format_prompt(prompt: str) -> str:
    """
    Formats the user's question into the Alpaca-style prompt template
    that the model was fine-tuned on. This is critical for getting accurate
    and relevant responses.

    Args:
        prompt: The raw question from the user.

    Returns:
        A string containing the fully formatted prompt.
    """
    return (
        f"### Instruction:\n"
        f"You are a legal expert. Provide a detailed and clear response to the legal question.\n\n"
        f"### Question:\n{prompt}\n\n"
        f"### Answer:"
    )

@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """
    API endpoint to handle legal questions. It receives a prompt,
    formats it, generates a response from the model, and returns it.
    """
    try:
        # Get the JSON data from the POST request.
        data = request.get_json()
        if not data:
            return jsonify({'response': "Error: Invalid JSON."}), 400
            
        prompt = data.get('prompt', '').strip()

        # Check if the prompt is empty.
        if not prompt:
            return jsonify({'response': "Error: Empty prompt."}), 400

        # Format the prompt and tokenize it.
        formatted_prompt = format_prompt(prompt)
        print(f"Formatted Prompt:\n{formatted_prompt}")
        
        # Note: We don't need .to(device) here because device_map="auto" handles it
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Generate the response from the model without calculating gradients.
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids.to(device), # Ensure input tensors are on the correct device for generation
                max_new_tokens=512,  # Maximum number of new tokens to generate.
                temperature=0.7,     # Controls randomness. Lower is more deterministic.
                top_p=0.9,           # Nucleus sampling: considers tokens with cumulative probability >= top_p.
                do_sample=True,      # Enable sampling to get more creative answers.
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated token IDs back into text.
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the answer part of the response.
        answer = generated_text.split("### Answer:")[-1].strip()
        print(f"Generated Answer:\n{answer}")

        # Return the answer as a JSON response.
        return jsonify({'response': answer})

    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({'response': f"An internal error occurred: {str(e)}"}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    API endpoint to handle PDF file uploads. It extracts text from the
    first few pages of the PDF.
    """
    # Check if a file named 'pdf' is in the request.
    if 'pdf' not in request.files:
        return jsonify({'text': 'No PDF file part in the request.'}), 400

    file = request.files['pdf']
    
    # Check if a file was actually selected.
    if file.filename == '':
        return jsonify({'text': 'No PDF file selected.'}), 400

    # Ensure the file is a PDF.
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'text': 'Invalid file type. Please upload a PDF.'}), 400

    try:
        text = ''
        # Open the PDF file using pdfplumber.
        with pdfplumber.open(file) as pdf:
            # Limit extraction to the first 3 pages to manage processing time.
            num_pages = min(len(pdf.pages), 3) 
            for page in pdf.pages[:num_pages]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        
        if not text:
             return jsonify({'text': 'Could not extract any text from the PDF.'})

        return jsonify({'text': text.strip()})
        
    except Exception as e:
        print(f"PDF parsing error: {e}")
        return jsonify({'text': f"Error processing PDF: {str(e)}"}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the Flask app.
    # host='0.0.0.0' makes the server accessible on your local network.
    # port=5000 is the standard port for Flask development.
    # debug=True enables auto-reloading and detailed error pages (disable for production).
    app.run(host='0.0.0.0', port=5000, debug=True)
