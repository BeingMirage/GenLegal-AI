import { useState } from 'react';
import axios from 'axios';
import './index.css';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async () => {
    const res = await axios.post('http://localhost:5000/api/prompt', { prompt });
    setResponse(res.data.response);
  };

  const handlePdfUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const formData = new FormData();
      formData.append('pdf', file);

      const res = await axios.post('http://localhost:5000/api/upload-pdf', formData);
      const extractedText = res.data.text;
      setPrompt((prev) => prev + '\n\n' + extractedText);
    }
  };

  return (
    <div className="genlegal-app">
      <aside className="info-panel">
        <h2 className="title">GenLegal</h2>
        <p className="description">
          An AI-powered legal assistant that helps you draft responses, understand legal clauses, and get insights — instantly. Built for lawyers, students, and researchers.
        </p>
      </aside>

      <main className="main-area">
        <h1>Hello, User</h1>
        <div className="response-box">{response}</div>

        <div className="input-bar">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask a legal question..."
          />

          <label className="pdf-upload-btn">
            📄
            <input type="file" accept=".pdf" onChange={handlePdfUpload} hidden />
          </label>

          <button onClick={handleSubmit}>▶</button>
        </div>
      </main>
    </div>
  );
}

export default App;
