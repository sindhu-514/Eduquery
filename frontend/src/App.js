import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { v4 as uuidv4 } from 'uuid';
import './styles/App.css';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import HomePage from './components/HomePage';

function MainApp() {
  const [books, setBooks] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedBook, setSelectedBook] = useState('');
  const [sessionId] = useState(uuidv4());
  const [uploadStatus, setUploadStatus] = useState('');
  const [endSessionStatus, setEndSessionStatus] = useState('');
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = () => {
    fetch("http://localhost:8007/books/")
      .then(res => res.json())
      .then(data => setBooks(data.books))
      .catch(err => console.error("Error fetching books:", err));
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.includes('pdf')) {
      setUploadStatus('⚠️ Please select a PDF file');
      return;
    }

    setIsLoading(true);
    setUploadStatus('📤 Uploading...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8007/upload/', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus('✅ ' + result.message);
        fetchBooks();
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        setUploadStatus('⚠️ ' + result.detail);
      }
    } catch (error) {
      setUploadStatus('⚠️ Upload failed: ' + error.message);
    } finally {
      setIsLoading(false);
      setTimeout(() => setUploadStatus(''), 5000);
    }
  };

  const sendMessage = () => {
    if (!selectedBook || !question.trim()) return;

    const userMessage = { sender: 'user', text: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append('query', question);
    formData.append('session_id', sessionId);
    formData.append('book_name', selectedBook);

    fetch("http://localhost:8007/query/", {
      method: 'POST',
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        const botResponse = { sender: 'bot', text: data.answer };
        setMessages((prev) => [...prev, botResponse]);
        setIsLoading(false);
      })
      .catch((err) => {
        setMessages((prev) => [
          ...prev,
          { sender: 'bot', text: '⚠️ Error fetching response.' },
        ]);
        setIsLoading(false);
      });
  };

  const clearChat = () => {
    setMessages([]);
  };

  const endSession = async () => {
    if (!selectedBook) {
      setEndSessionStatus('⚠️ No book selected.');
      return;
    }
    setIsLoading(true);
    setEndSessionStatus('Ending session...');
    const formData = new FormData();
    formData.append('session_id', sessionId);
    formData.append('book_name', selectedBook);
    try {
      const response = await fetch('http://localhost:8007/end_session/', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        setEndSessionStatus('✅ ' + result.message);
        setMessages([]);
        setSelectedBook('');
        fetchBooks();
      } else {
        setEndSessionStatus('⚠️ ' + (result.message || 'Failed to end session.'));
      }
    } catch (error) {
      setEndSessionStatus('⚠️ Failed to end session: ' + error.message);
    } finally {
      setIsLoading(false);
      setTimeout(() => setEndSessionStatus(''), 5000);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">📘 EduQuery</div>
        <div className="header-controls">
          <div className="upload-section">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
              ref={fileInputRef}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="upload-button"
              disabled={isLoading}
              style={{
                padding: '6px 12px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                marginRight: '10px'
              }}
            >
              📎 Upload PDF
            </button>
            {uploadStatus && (
              <span className="upload-status" style={{ marginRight: '10px' }}>
                {uploadStatus}
              </span>
            )}
          </div>
          <select
            value={selectedBook}
            onChange={(e) => setSelectedBook(e.target.value)}
            className="book-select"
          >
            <option value="">Select a book</option>
            {books.map((book, idx) => (
              <option key={idx} value={book}>
                {book}
              </option>
            ))}
          </select>
          {selectedBook && (
            <button
              onClick={endSession}
              className="end-session-button"
              disabled={isLoading}
              style={{
                padding: '6px 12px',
                backgroundColor: '#d9534f',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                marginLeft: '10px'
              }}
            >
              🚪 End Session
            </button>
          )}
          {endSessionStatus && (
            <span className="end-session-status" style={{ marginLeft: '10px' }}>
              {endSessionStatus}
            </span>
          )}
        </div>
      </header>

      <div className="main-content">
        <div className="pdf-panel">
          {selectedBook ? (
            <iframe
              src={`http://localhost:8007/book?subject=${selectedBook}`}
              className="pdf-viewer"
              title="PDF Viewer"
            />
          ) : (
            <div className="pdf-placeholder">Select a book to start reading</div>
          )}
        </div>

        <div className="chat-panel">
          <div className="chat-header">
            💬 Ask EduQuery
            {selectedBook && (
              <button
                style={{
                  marginLeft: '10px',
                  padding: '6px 12px',
                  backgroundColor: '#ff5c5c',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
                onClick={clearChat}
              >
                🗑️ Clear Chat
              </button>
            )}
          </div>

          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.sender}`}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            ))}
            {isLoading && (
              <div className="chat-message bot loading">
                <div className="loading-spinner"></div>
              </div>
            )}
          </div>

          <div className="chat-input">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={selectedBook ? 'Type your question...' : 'Select a book first'}
              disabled={!selectedBook || isLoading}
            />
            <button onClick={sendMessage} disabled={!selectedBook || isLoading}>
              📤
            </button>
          </div>
        </div>
      </div>

      <footer className="app-footer">
        EduQuery: Ask questions about the PDF document
      </footer>
    </div>
  );
}

function App() {
  const navigate = useNavigate();
  return (
    <Routes>
      <Route path="/" element={<HomePage onGetStarted={() => navigate('/app')} />} />
      <Route path="/app" element={<MainApp />} />
    </Routes>
  );
}

export default function AppWithRouter() {
  return (
    <Router>
      <App />
    </Router>
  );
}