# GenLegal-AI

GenLegal-AI is a legal question-answering application leveraging modern AI and web technologies.

## Tech Stack

- **Frontend:** React (TypeScript, Vite)
- **Backend:** Python (Flask)
- **AI Model/API:** Google Gemini API (gemini-2.5-flash)
- **Other:** Node.js (for frontend tooling)

## How to Run the Project Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BeingMirage/GenLegal-AI.git
   cd GenLegal-AI
   ```

2. **Backend Setup:**
   - Create and activate a Python virtual environment:
     ```bash
     python -m venv venv
     # On Unix/macOS:
     source venv/bin/activate
     # On Windows:
     venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Set your Google Gemini API key in the backend (currently hardcoded in `backend/app.py` for demo purposes).
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

- **Google Gemini API:** The backend uses the Google Gemini API (model: `gemini-2.5-flash`) to generate answers to legal questions. All inference is performed via calls to this external API.
- **Why Gemini?** Gemini is a state-of-the-art generative language model capable of providing detailed, context-aware answers. Using Gemini allows leveraging powerful language understanding without the need for local model hosting or fine-tuning.

## Assumptions & Limitations

- The system relies on the Google Gemini API for all legal Q&A responses.
- Answers are generated based on the Gemini model’s general knowledge and may not always be fully accurate or up-to-date with the latest legal developments.
- The model is prompted to focus on Indian law, but its responses may sometimes lack jurisdictional specificity or legal nuance.
- The system does not provide professional legal advice; it is for informational purposes only.
- Requires a valid Google Gemini API key and internet connectivity for backend operation.

---
