# 📚 EduQuery: Intelligent PDF-Based Educational Assistant

EduQuery is a personalized learning platform that helps users interact with educational textbooks using AI-powered Q&A. It simplifies complex textbook content into easy-to-understand answers and makes study more engaging and accessible.

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

- **Ollama + LLaMA3**  
  Used for local embedding and inference. Fast and context-aware.

- **Mistral**  
  Lightweight, efficient model with decent summarization — used for comparison.

- **Gemini (Google)**  
  Eventually chosen as the main model for final response generation due to:
  - Superior language fluency
  - Better simplification of complex academic content
  - Stronger reasoning and follow-up capabilities

We use **LLaMA3 + `bge-m3`** via Ollama for embedding the textbook content and **ChromaDB** for vector similarity search.

---

## ⚙️ Tech Stack

- **FastAPI** (Backend API)
- **Ollama** (Local LLM & Embedding)
- **ChromaDB** (Vector DB for storing PDF embeddings)
- **LangChain** (Prompt handling & retrieval logic)
- **Gemini** (Final LLM response generation)
- *(Frontend in progress: React-based viewer and chat interface)*

---

## 📦 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/sindhu-514/Eduquery.git
cd Eduquery/try1/src
