# main.py
# This script creates a Flask web server that interacts with the Google Gemini API.

import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- API Token Setup ---
# Your Google Gemini API key is hardcoded here.
# IMPORTANT: For production, it's highly recommended to use environment variables
# instead of hardcoding keys directly in the code.
API_TOKEN = "AIzaSyBvRGpfzTIp1itCP1vrwrkbU7ruJG248a8" 

# --- Configure Gemini API ---
try:
    genai.configure(api_key=API_TOKEN)
    # Initialize the Gemini Pro model
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("Google Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """
    API endpoint that receives a prompt, sends it to the Google Gemini API,
    and returns the model's response.
    """
    if not model:
        return jsonify({'response': "Error: Gemini API is not configured."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': "Error: Invalid JSON."}), 400

        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'response': "Error: Empty prompt."}), 400

        # Create a detailed prompt that gives the model a role and clear instructions.
        # This works very well with powerful models like Gemini.
        system_prompt = (
            "You are an expert legal assistant specializing in Indian law. "
            "Your task is to provide a detailed, comprehensive, and well-explained answer "
            "to the following question. Ensure you cover the main points, provide relevant "
            "context, and cite sections or articles of the Indian Constitution or relevant laws if applicable."
        )

        full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nDetailed Answer:"

        print(f"Sending prompt to Gemini API...")

        # Send the prompt to the Gemini API
        response = model.generate_content(full_prompt)

        # Extract the text from the response
        answer = response.text

        print(f"Received answer from Gemini API.")
        return jsonify({'response': answer.strip()})

    except Exception as e:
        print(f"Error during API call: {e}")
        return jsonify({'response': f"An internal error occurred: {str(e)}"}), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    This endpoint for PDF processing remains unchanged.
    """
    if 'pdf' not in request.files:
        return jsonify({'text': 'No PDF file part in the request.'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'text': 'No PDF file selected.'}), 400

    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'text': 'Invalid file type. Please upload a PDF.'}), 400

    try:
        text = ''
        with pdfplumber.open(file) as pdf:
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
    app.run(host='0.0.0.0', port=5000, debug=True)
