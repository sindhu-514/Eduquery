# 📚 EduQuery: Intelligent PDF-Based Educational Assistant

EduQuery is a personalized learning platform that helps users interact with educational textbooks using AI-powered Q&A. It simplifies complex textbook content into easy-to-understand answers and makes study more engaging and accessible.

## 🖥️ Application Screenshots

### Welcome Screen
![EduQuery Welcome Screen](./images/home.png)
*The welcoming interface that introduces users to EduQuery's AI-powered PDF learning capabilities*

### Main Application Interface
![EduQuery Main Interface](./images/main.png)
*The main application showing the split-screen layout with PDF viewer and chat interface*

### Active Learning Session
![EduQuery Active Session](./images/example.png)
*An active learning session where users can upload PDFs, view content, and ask questions to get AI-powered answers*

---

## 🚀 How It Works

1. **Upload a Book**  
   Users can upload their educational textbooks in PDF format.

2. **Book Selection Dropdown**  
   Uploaded books are displayed in a dropdown. When a user selects a book, the PDF is rendered and viewable inside the application.

3. **Instant Doubt Solving**  
   While reading, if a user has a doubt, they can immediately ask the built-in **AI chatbot**.  
   The chatbot retrieves the answer from the selected textbook and responds in a simplified, student-friendly way.

---

## 🧠 AI Models Used

We experimented with multiple models to optimize retrieval and answer quality:

| Model | Contextual Accuracy | Behavior |
|-------|-------------------|----------|
| **Gemini** | 🟢 **Excellent** – Structured and coherent | Extracts relevant, document-aligned insights |
| **Llama3** | 🟠 Moderate – Occasionally off-topic | Drifts from topic; includes inferred content |
| **Mistral** | 🔴 Poor – Fragmented and vague | Struggles with complex input; low consistency |

### **Recommended: Gemini (Google)**
**Gemini** is our **primary and recommended model** for final response generation due to:
- Superior language fluency and coherence
- Better simplification of complex academic content
- Stronger reasoning and follow-up capabilities
- Excellent contextual accuracy and document alignment

### **Embedding Model**
We use **`bge-m3`** via Ollama for embedding the textbook content and **ChromaDB** for vector similarity search.

---

## ⚙️ Tech Stack

- **FastAPI** (Backend API)
- **Ollama** (Local LLM & Embedding)
- **ChromaDB** (Vector DB for storing PDF embeddings)
- **LangChain** (Prompt handling & retrieval logic)
- **Gemini** (Final LLM response generation)
- **React** (Frontend viewer and chat interface)

---

## 📋 Prerequisites

Before running EduQuery, make sure you have the following installed:

- **Python 3.8+**
- **Node.js 16+** and **npm**
- **Ollama** (for local LLM processing)
- **Google Gemini API Key** (for final response generation)

### Installing Prerequisites

1. **Install Ollama:**
   ```bash
   # On Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows
   # Download from https://ollama.ai/download
   ```

2. **Pull required Ollama models:**
   ```bash
   # Required for embeddings (used by main application)
   ollama pull bge-m3:latest
   
   # Optional: Only needed for experimental model testing
   ollama pull llama3:8b
   ollama pull mistral:7b
   ```

3. **Get Google Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Save it for later use

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Aishwarya2k5p/EduQuery.git
cd Eduquery
```

### 2. Install Requirements

#### Backend Requirements
```bash
# Create virtual environment (from /Eduquery directory)
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies (requirements.txt is in /Eduquery)
pip install -r requirements.txt

# Navigate to backend directory
cd backend

# Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env

# Initialize directories
python -c "from EduQuery.config import Config; Config.initialize()"
```

#### Frontend Requirements
```bash
# Navigate to frontend directory (from /Eduquery)
cd frontend

# Install Node.js dependencies
npm install
```

---

## 🚀 Running the Application

### Step 1: Start the Backend Server
```bash
# Navigate to backend directory (from /Eduquery)
cd backend

# Activate virtual environment (if not already activated)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Start the FastAPI server using uvicorn
uvicorn EduQuery.main:app --host 0.0.0.0 --port 8006 --reload
```

The backend will start on `http://localhost:8006`

### Step 2: Start the Frontend Development Server
```bash
# In a new terminal, navigate to frontend directory (from /Eduquery)
cd frontend

# Start the React development server
npm start
```

The frontend will start on `http://localhost:3000`

### Option 2: Experimental Model Versions (Not Recommended)

The project includes experimental implementations for comparison purposes:

> **⚠️ Note**: These are test implementations and not recommended for production use. The main application uses Gemini for optimal performance.

#### Experimental: LLaMA3 Implementation
```bash
# From /Eduquery directory
cd backend
source venv/bin/activate
python llama3.py
```

#### Experimental: Mistral Implementation
```bash
# From /Eduquery directory
cd backend
source venv/bin/activate
python mistral.py
```

---

## 🔧 Configuration

### Backend Configuration
- **Port**: Default is 8006 (configurable in `backend/EduQuery/config.py`)
- **PDF Directory**: `./books` (auto-created)
- **ChromaDB Path**: `./data/chroma_db` (auto-created)

### Frontend Configuration
- **Port**: Default is 3000
- **API Endpoint**: Configured to connect to backend on port 8006

### Environment Variables
Create a `.env` file in the `backend` directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 📁 Project Structure

```
Eduquery/
├── requirements.txt           # Python dependencies (root level)
├── backend/                    # 🚀 MAIN BACKEND APPLICATION
│   ├── EduQuery/
│   │   ├── __init__.py
│   │   ├── main.py              # Main FastAPI application (Gemini-based)
│   │   ├── api_routes.py        # API endpoints
│   │   ├── pdf_processor.py     # PDF processing logic
│   │   ├── query_processor.py   # Query handling
│   │   ├── chat_manager.py      # Chat history management
│   │   ├── embedder.py          # Embedding generation
│   │   ├── llm.py              # LLM integration (Gemini)
│   │   ├── utils.py            # Utility functions
│   │   └── config.py           # Configuration
│   ├── .env                    # Environment variables
│   ├── llama3.py              # 🔬 EXPERIMENTAL: LLaMA3 test implementation
│   └── mistral.py             # 🔬 EXPERIMENTAL: Mistral test implementation
├── frontend/                   # 🚀 MAIN FRONTEND APPLICATION
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── api/               # API integration
│   │   └── styles/            # CSS styles
│   ├── package.json           # Node.js dependencies
│   └── public/                # Static assets
└── README.md
```

> **🎯 Main Application**: Use `backend/EduQuery/main.py` and `frontend/` for the production-ready Gemini-based application.
> 
> **🔬 Experimental Files**: `llama3.py` and `mistral.py` are test implementations for model comparison only.

---

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not running:**
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Port already in use:**
   - Change port in `backend/EduQuery/config.py`
   - Or kill process using the port

3. **Missing dependencies:**
   ```bash
   # Backend (from /Eduquery directory)
   pip install -r requirements.txt
   
   # Frontend
   npm install
   ```

4. **API key issues:**
   - Ensure `.env` file exists in backend directory
   - Verify Gemini API key is valid

---

## 📝 Usage

1. **Start both servers** (backend and frontend)
2. **Open browser** to `http://localhost:3000`
3. **Upload a PDF** textbook using the upload button
4. **Select the book** from the dropdown
5. **Ask questions** in the chat interface
6. **Get instant answers** from your textbook content
7. **End session** when finished:
   - Click the "🚪 End Session" button to remove the current book
   - This will clear the chat history and remove the book from the system
   - You can then upload a new book or select a different one

---

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests


