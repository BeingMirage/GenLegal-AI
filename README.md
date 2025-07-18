# GenLegal-AI

GenLegal-AI is a legal question-answering application leveraging modern AI and web technologies.

## Tech Stack

- **Frontend:** React (TypeScript, Vite)
- **Backend:** Python (Flask)
- **AI Model:** Fine-tuned FLAN-T5 (with LoRA adapters)
- **Other:** Node.js (for frontend tooling), PyTorch/Transformers (for model training/inference)

## How to Run the Project Locally

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd GenLegal-AI
   ```

2. **Backend Setup:**
   - Create and activate a Python virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Start the backend server:
     ```bash
     cd backend
     python app.py
     ```

3. **Frontend Setup:**
   - Install Node.js dependencies:
     ```bash
     cd ../frontend
     npm install
     ```
   - Start the frontend development server:
     ```bash
     npm run dev
     ```
   - The app will be available at `http://localhost:5173` (or as indicated in the terminal).

## API Used

- **Model API:** The backend serves a fine-tuned FLAN-T5 model (using LoRA adapters) for legal question answering. No external APIs are used for inference; all model logic runs locally.
- **Why FLAN-T5?** FLAN-T5 is a powerful, instruction-tuned language model, and LoRA adapters allow efficient fine-tuning for domain-specific tasks (like legal Q&A) without retraining the entire model.

## Assumptions & Limitations

- The legal AI model is trained on a limited set of legal Q&A data (e.g., Indian law datasets like IPC, CrPC, Constitution).
- The model may not generalize to all legal domains or jurisdictions.
- The system does not provide legal advice; it is for informational purposes only.
- Model inference may require a GPU for reasonable performance.
- The frontend and backend must be run separately in development.

---

Feel free to modify this README to better fit your deployment or add more details as needed!
