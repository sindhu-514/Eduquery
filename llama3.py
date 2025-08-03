"""
Optimized Llama3-based EduQuery RAG System
Combined implementation with performance enhancements for fast responses
"""

import os
import logging
import time
import hashlib
import pickle
import tempfile
import shutil
import re
import numpy as np
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from chromadb import PersistentClient
from uvicorn import run
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    CHROMA_DB_PATH = "./data/chroma_db"
    EMBEDDING_MODEL_NAME = "bge-m3:latest"  # Optimized for Llama3
    LLAMA_MODEL_NAME = "llama3:8b"  # Fast and efficient
    FASTAPI_PORT = 8008  # Different port to avoid conflicts
    PDF_DIR = "./books"
    CACHE_DIR = "./cache"
    MAX_CACHE_SIZE = 1000
    BATCH_SIZE = 64  # Optimized batch size for Llama3
    MAX_WORKERS = max(1, cpu_count() - 1)
    OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
    LLAMA_ENDPOINT = "http://localhost:11434/api/generate"
    
    # Llama3-specific optimizations
    LLAMA_TEMPERATURE = 0.3  # Balanced temperature for Llama3
    LLAMA_TOP_P = 0.9
    LLAMA_TOP_K = 40
    LLAMA_REPEAT_PENALTY = 1.1
    LLAMA_MAX_TOKENS = 2048  # Optimized for Llama3
    
    @classmethod
    def initialize(cls):
        os.makedirs(cls.PDF_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.CHROMA_DB_PATH), exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

# ============================================================================
# CACHING SYSTEM
# ============================================================================

class OptimizedCache:
    def __init__(self, cache_dir: str, max_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_file = os.path.join(cache_dir, "llama3_cache.pkl")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> OrderedDict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    return OrderedDict(data) if isinstance(data, dict) else OrderedDict()
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        return OrderedDict()

    def get(self, key: str) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, response: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = response

        if len(self.cache) > self.max_size:
            removed_key, _ = self.cache.popitem(last=False)
            logging.info(f"Evicted cache key: {removed_key[:8]}...")

        self._save_cache()

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

# ============================================================================
# OPTIMIZED EMBEDDER FOR LLAMA3
# ============================================================================

class OptimizedLlama3Embedder:
    def __init__(self, model_name=Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.endpoint = Config.OLLAMA_ENDPOINT
        self.session = requests.Session()
        self.session.timeout = 15  # Faster timeout for Llama3
    
    def embed_query(self, text: str) -> List[float]:
        """Fast single query embedding"""
        try:
            response = self.session.post(
                self.endpoint,
                json={"model": self.model_name, "prompt": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return self._get_zero_embedding()
    
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """Optimized batch embedding for Llama3"""
        if not texts:
            return []
        
        logging.info(f"Generating embeddings for {len(texts)} documents")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent embeddings
        embeddings = [None] * len(texts)
        max_workers = min(Config.MAX_WORKERS, len(texts), 8)  # Limit for stability
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._embed_with_retry, text): i 
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[index] = embedding
                except Exception as e:
                    logging.error(f"Failed to embed text {index}: {e}")
                    embeddings[index] = self._get_zero_embedding()
        
        elapsed = time.time() - start_time
        logging.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        return embeddings

    def _embed_with_retry(self, text: str, max_retries: int = 2) -> List[float]:
        """Embed with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.embed_query(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.1 * (attempt + 1))

    def _get_zero_embedding(self) -> List[float]:
        return [0.0] * 1024  # bge-m3 dimension

    def check_pdf_similarity(self, uploaded_file_path: str, collection, similarity_threshold=0.95):
        """Check for duplicate PDFs"""
        try:
            loader = PyPDFLoader(uploaded_file_path)
            pages = loader.load()
            sample_pages = pages[:min(3, len(pages))]
            sample_texts = [page.page_content.strip() for page in sample_pages if page.page_content.strip()]
            
            if not sample_texts:
                return False, None
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            documents = [Document(page_content=text) for text in sample_texts]
            chunks = text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]
            
            sample_embeddings = self.embed_documents_batch(chunk_texts)
            
            existing_results = collection.get()
            if not existing_results['embeddings']:
                return False, None
            
            existing_embeddings = existing_results['embeddings']
            existing_metadatas = existing_results['metadatas']
            
            source_groups = {}
            for i, metadata in enumerate(existing_metadatas):
                source = metadata.get('source', 'unknown')
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(existing_embeddings[i])
            
            for source_name, source_embeddings in source_groups.items():
                max_similarity = 0
                for sample_emb in sample_embeddings:
                    for existing_emb in source_embeddings:
                        similarity = np.dot(sample_emb, existing_emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(existing_emb))
                        max_similarity = max(max_similarity, similarity)
                
                if max_similarity >= similarity_threshold:
                    return True, source_name
            
            return False, None
            
        except Exception as e:
            logging.error(f"Error checking PDF similarity: {e}")
            return False, None

# ============================================================================
# OPTIMIZED LLAMA3 LLM
# ============================================================================

class OptimizedLlama3LLM:
    def __init__(self):
        self.model_name = Config.LLAMA_MODEL_NAME
        self.cache = OptimizedCache(Config.CACHE_DIR, Config.MAX_CACHE_SIZE)
        self.session = requests.Session()
        self.session.timeout = 60
    
    def get_cache_key(self, question: str, book_name: str, retrieved_texts: str) -> str:
        """Generate cache key"""
        content = f"{question.strip()}||{book_name.strip()}||{retrieved_texts.strip()}"
        return hashlib.md5(content.encode()).hexdigest()

    def invoke(self, prompt: str, question: str = None, book_name: str = None, retrieved_texts: str = None) -> Any:
        """Optimized Llama3 invocation with caching"""
        try:
            # Check cache first
            cache_key = prompt
            if question and book_name and retrieved_texts:
                cache_key = self.get_cache_key(question, book_name, retrieved_texts)
            
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return type('Obj', (object,), {'content': cached_response})
            
            # Check if Llama3 model is available
            try:
                models_response = self.session.get("http://localhost:11434/api/tags", timeout=5)
                models_response.raise_for_status()
                available_models = models_response.json().get("models", [])
                model_names = [model.get("name", "") for model in available_models]
                
                logging.info(f"Available models: {model_names}")
                
                if self.model_name not in model_names:
                    logging.error(f"Llama3 model '{self.model_name}' not found. Available models: {model_names}")
                    return type('Obj', (object,), {'content': f"Error: Llama3 model '{self.model_name}' is not available. Please run 'ollama pull {self.model_name}' first."})
                    
            except Exception as e:
                logging.error(f"Could not check available models: {e}")
                return type('Obj', (object,), {'content': "Error: Could not connect to Ollama. Please ensure Ollama is running with 'ollama serve'"})
            
            # Generate response with optimized settings for Llama3
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": Config.LLAMA_TEMPERATURE,
                    "top_p": Config.LLAMA_TOP_P,
                    "top_k": Config.LLAMA_TOP_K,
                    "repeat_penalty": Config.LLAMA_REPEAT_PENALTY,
                    "num_predict": Config.LLAMA_MAX_TOKENS,
                    "num_ctx": 4096,
                    "num_thread": 4
                }
            }
            
            logging.info(f"Generating response with Llama3 (timeout: 60s)...")
            response = self.session.post(Config.LLAMA_ENDPOINT, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            response_text = result.get("response", "No response generated")
            logging.info(f"Llama3 response generated successfully (length: {len(response_text)})")
            
            # Cache the response
            self.cache.set(cache_key, response_text)
            
            return type('Obj', (object,), {'content': response_text})
            
        except Exception as e:
            logging.error(f"Llama3 generation error: {e}")
            if "404" in str(e):
                return type('Obj', (object,), {'content': f"Error: Llama3 model '{self.model_name}' not found. Please run 'ollama pull {self.model_name}' to install it."})
            elif "Connection" in str(e):
                return type('Obj', (object,), {'content': "Error: Could not connect to Ollama. Please ensure Ollama is running with 'ollama serve'"})
            else:
                return type('Obj', (object,), {'content': f"Error generating response: {str(e)}"})

# ============================================================================
# PDF PROCESSOR
# ============================================================================

def process_pdf_page(args):
    """Process single PDF page"""
    pdf_path, page_num = args
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if page_num < len(pages):
            page = pages[page_num]
            text = page.page_content.strip()
            if text:
                return {
                    'content': text,
                    'page_number': page_num,
                    'source': os.path.basename(pdf_path)
                }
    except Exception as e:
        logging.error(f"Error processing page {page_num} of {pdf_path}: {e}")
    return None

class OptimizedPDFProcessor:
    def __init__(self, collection):
        self.collection = collection
        self.embedder = OptimizedLlama3Embedder()
    
    def extract_pdf_text_parallel(self, pdf_path: str) -> List[Dict]:
        """Parallel PDF text extraction"""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            num_pages = len(pages)
            logging.info(f"Processing {num_pages} pages in parallel for {pdf_path}")
            
            args = [(pdf_path, i) for i in range(num_pages)]
            with Pool(processes=min(Config.MAX_WORKERS, num_pages)) as pool:
                results = pool.map(process_pdf_page, args)
            
            processed_pages = [page for page in results if page is not None]
            logging.info(f"Successfully processed {len(processed_pages)} pages from {pdf_path}")
            return processed_pages
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")

    def split_into_chunks(self, pages: List[Dict]) -> List[Document]:
        """Split pages into optimized chunks"""
        logging.info("Starting chunking of extracted pages...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100
        )
        all_texts = [page['content'] for page in pages]
        documents = [Document(page_content=text) for text in all_texts]
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Chunking complete: {len(chunks)} chunks created.")
        return chunks

    def generate_embeddings(self, chunks: List[Document]) -> Tuple[List[str], List[List[float]]]:
        """Generate embeddings for chunks"""
        chunk_texts = [chunk.page_content for chunk in chunks]
        logging.info(f"Starting embedding for {len(chunk_texts)} chunks...")
        chunk_embeddings = self.embedder.embed_documents_batch(chunk_texts)
        logging.info(f"Embedding complete: {len(chunk_embeddings)} embeddings generated.")
        return chunk_texts, chunk_embeddings

    def store_embeddings(self, chunk_texts: List[str], chunk_embeddings: List[List[float]], 
                        metadatas: List[Dict], ids: List[str]):
        """Store embeddings in ChromaDB with duplicate handling"""
        try:
            # Check if embeddings already exist
            existing_ids = set()
            try:
                existing_results = self.collection.get(where={"source": metadatas[0]["source"]})
                existing_ids = set(existing_results.get("ids", []))
            except Exception:
                pass
            
            # Filter out existing chunks
            new_chunk_texts = []
            new_chunk_embeddings = []
            new_metadatas = []
            new_ids = []
            
            for i, chunk_id in enumerate(ids):
                if chunk_id not in existing_ids:
                    new_chunk_texts.append(chunk_texts[i])
                    new_chunk_embeddings.append(chunk_embeddings[i])
                    new_metadatas.append(metadatas[i])
                    new_ids.append(chunk_id)
            
            if new_chunk_texts:
                self.collection.add(
                    documents=new_chunk_texts,
                    embeddings=new_chunk_embeddings,
                    metadatas=new_metadatas,
                    ids=new_ids
                )
                logging.info(f"Successfully added {len(new_chunk_texts)} new chunks to ChromaDB")
            else:
                logging.info(f"All chunks for {metadatas[0]['source']} already exist in ChromaDB")
            
            count = self.collection.count()
            logging.info(f"Total chunks in collection: {count}")
            
        except Exception as e:
            logging.error(f"Error storing embeddings: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> bool:
        """Complete PDF processing pipeline"""
        pdf_name = os.path.basename(pdf_path)
        logging.info(f"=== STARTING PDF PROCESSING: {pdf_name} ===")
        start_time = time.time()
        
        try:
            pages = self.extract_pdf_text_parallel(pdf_path)
            logging.info(f"Extracted {len(pages)} pages from PDF")
            
            chunks = self.split_into_chunks(pages)
            chunk_texts, chunk_embeddings = self.generate_embeddings(chunks)
            
            ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunk_texts))]
            metadatas = [{
                "source": pdf_name,
                "content_type": "text",
                "page_number": pages[i % len(pages)]['page_number'] if pages else 0
            } for i in range(len(chunk_texts))]
            
            self.store_embeddings(chunk_texts, chunk_embeddings, metadatas, ids)
            
            processing_time = time.time() - start_time
            logging.info(f"=== COMPLETED PDF PROCESSING: {pdf_name} in {processing_time:.2f}s ===")
            return True
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# ============================================================================
# CHAT MANAGER
# ============================================================================

class OptimizedChatManager:
    def __init__(self):
        self.chat_histories = {}
    
    def get_history(self, session_id: str) -> List[Dict]:
        return self.chat_histories.get(session_id, [])
    
    def add_to_history(self, session_id: str, question: str, answer: str, embedding: List[float] = None):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        
        self.chat_histories[session_id].append({
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "qa_text": f"Q: {question}\nA: {answer}"
        })
    
    def get_recent_history(self, session_id: str, limit: int = 5) -> List[Dict]:
        history = self.get_history(session_id)
        return history[-limit:]

    def get_hybrid_history(self, session_id: str, query_embedding: List[float], 
                          recent_n: int = 5, relevant_k: int = 2) -> List[Dict]:
        history = self.get_history(session_id)
        if not history:
            return []
        
        recent = history[-recent_n:]
        candidates = [h for h in history if h.get("embedding") is not None and h not in recent]
        
        if not candidates or query_embedding is None:
            return recent
        
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        
        scored = [(h, cosine_sim(query_embedding, h["embedding"])) for h in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        relevant = [h[0] for h in scored[:relevant_k] if h[1] > 0.85]
        
        return recent + relevant

# ============================================================================
# UTILITIES
# ============================================================================

CASUAL_RESPONSES = {
    "hello": "Hello! How can I assist you with your PDF content today?",
    "hi": "Hi there! Ready to help with your queries.",
    "hey": "Hey! Ask me anything related to the document.",
    "thanks": "You're welcome!",
    "thank you": "Glad I could help!",
    "who are you": "I'm an educational assistant designed to answer questions from your PDF content.",
    "good morning": "Good morning! Let me know what you'd like to learn today.",
    "good afternoon": "Good afternoon! How can I help you with the document?",
    "good evening": "Good evening! Ready to assist with your PDF queries.",
    "bye": "Goodbye! Feel free to return if you have more questions.",
    "goodbye": "Take care! Come back anytime for more help.",
    "see you": "See you later! Have a great day.",
    "how are you": "I'm doing well, thank you! How can I help you with your PDF content?",
    "what can you do": "I can answer questions about your uploaded PDF documents, help you understand the content, and provide detailed explanations. Just ask me anything related to the document!",
    "help": "I'm here to help! You can ask me questions about your PDF content, request explanations, or ask for summaries. What would you like to know?",
}

def is_followup_query(query: str) -> bool:
    """Check if query is a follow-up"""
    obvious_followup_phrases = [
        'more', 'in short', 'briefly', 'summarize', 'key points', 'uses', 'examples',
        'explain about', 'tell me about', 'describe about', 'what about', 'how about',
        'in detail', 'in depth', 'more details', 'more information', 'give details',
        'explain them', 'tell me them', 'describe them', 'about them', 'them',
        'simple terms', 'simple', 'beginner', 'easy', 'basics'
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in obvious_followup_phrases)

def is_new_topic_query(query: str, recent_history: List[Dict]) -> bool:
    """Check if query is a new topic"""
    if not recent_history:
        return True
    
    def extract_terms(text):
        terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        common_words = {'what', 'is', 'are', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'this', 'that', 'these', 'those', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me', 'more', 'detail', 'explain', 'state', 'all', 'uses', 'use', 'simple', 'beginner', 'easy', 'basics', 'short', 'key', 'points', 'summary', 'them', 'they', 'their'}
        return terms - common_words
    
    current_terms = set(extract_terms(query.lower()))
    history_terms = set()
    
    for entry in recent_history[-5:]:
        question = entry.get('question', '').lower()
        answer = entry.get('answer', '').lower()
        history_terms.update(extract_terms(question))
        history_terms.update(extract_terms(answer))
    
    new_topic_indicators = [
        'what is', 'define', 'explain', 'tell me about', 'describe',
        'how does', 'what are', 'list', 'show me', 'give me',
    ]
    has_new_topic_indicators = any(indicator in query.lower() for indicator in new_topic_indicators)
    
    if current_terms and history_terms:
        overlap = len(current_terms.intersection(history_terms))
        total_unique = len(current_terms.union(history_terms))
        similarity = overlap / total_unique if total_unique > 0 else 0
        return similarity < 0.15 and has_new_topic_indicators
    
    return has_new_topic_indicators

def is_casual_query(query: str) -> bool:
    """Check if query is casual"""
    normalized_query = query.lower().strip()
    for keyword in CASUAL_RESPONSES.keys():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return True
    return False

def handle_casual_query(query: str) -> str:
    """Handle casual queries"""
    normalized_query = query.lower().strip()
    for keyword, response in CASUAL_RESPONSES.items():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return response
    return "Hello! Feel free to ask questions about the PDF content."

def rewrite_followup_query(current_query: str, chat_history: List[Dict]) -> str:
    """Rewrite follow-up queries"""
    if not is_followup_query(current_query):
        return current_query

    anchor_question = ""
    for entry in reversed(chat_history):
        q = entry.get("question", "")
        if q and not is_followup_query(q) and not is_casual_query(q):
            anchor_question = q.strip()
            break

    if anchor_question:
        combined_query = f"{anchor_question} {current_query}".strip()
        logging.info(f"[Follow-up] Rewriting: '{current_query}' => '{combined_query}' (anchor: '{anchor_question}')")
        return combined_query

    return current_query

# ============================================================================
# QUERY PROCESSOR
# ============================================================================

class OptimizedQueryProcessor:
    def __init__(self, collection, chat_manager: OptimizedChatManager):
        self.collection = collection
        self.chat_manager = chat_manager
        self.embedder = OptimizedLlama3Embedder()
        self.llm = OptimizedLlama3LLM()
        logging.info("Using Optimized Llama3 LLM with caching for responses.")

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)

    def semantic_search(self, query_embedding: List[float], book_name: str, n_results: int) -> Tuple[List[str], List[Dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"source": book_name}
        )
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        return documents, metadatas

    def handle_list_query_fallback(self, book_name: str) -> Tuple[List[str], List[Dict]]:
        all_results = self.collection.get(
            where={"source": book_name},
            include=["documents", "metadatas"]
        )
        all_docs = all_results.get("documents", [])
        all_metas = all_results.get("metadatas", [])
        return all_docs, all_metas

    def retrieve_relevant_content(self, query: str, book_name: str, n_results: int = 15) -> Tuple[List[str], List[Dict]]:
        try:
            LIST_KEYWORDS = ['list', 'all', 'every', 'complete', 'entire', 'phases', 'steps', 'process']
            query_lower = query.lower()
            is_list_query = any(keyword in query_lower for keyword in LIST_KEYWORDS)
            
            query_embedding = self.embed_query(query)
            
            if is_list_query:
                n_results = max(20, n_results)
            
            documents, metadatas = self.semantic_search(query_embedding, book_name, n_results)
            
            if not documents:
                return [], []
            
            if is_list_query:
                all_docs, all_metas = self.handle_list_query_fallback(book_name)
                if len(all_docs) > len(documents):
                    documents = all_docs
                    metadatas = all_metas
            
            logging.info(f"Retrieved {len(documents)} chunks for query: {query}")
            return documents, metadatas
            
        except Exception as e:
            logging.error(f"Error retrieving content: {e}")
            return [], []

    def build_optimized_prompt(self, query: str, retrieved_texts: str, chat_history: str = "") -> str:
        """Build optimized prompt for Llama3"""
        base_prompt = f"""
You are an educational assistant. Answer using ONLY the PDF content below. Be concise but comprehensive.

PDF Content:
{retrieved_texts}

Question: {query}

Answer:"""
        
        if chat_history:
            prompt = f"""
You are an educational assistant. Consider conversation context and answer using ONLY the PDF content.

Previous conversation:
{chat_history}

{base_prompt}

Answer:"""
        else:
            prompt = f"""
{base_prompt}

Answer:"""
        
        return prompt

    def generate_response(self, query: str, retrieved_texts: str, session_id: str, book_name: str = None) -> Tuple[str, Optional[List[float]]]:
        try:
            query_embedding = self.embed_query(query)
            is_followup = is_followup_query(query)
            recent_history = self.chat_manager.get_recent_history(session_id, limit=8)
            hybrid_history = self.chat_manager.get_hybrid_history(session_id, query_embedding, recent_n=3, relevant_k=2)
            
            if not is_followup:
                is_new_topic = is_new_topic_query(query, recent_history) if recent_history else True
                if is_new_topic:
                    hybrid_history = []
            
            history_text = ""
            if hybrid_history:
                history_text = "\n\n".join([
                    f"Q: {h['question']}\nA: {h['answer']}" for h in hybrid_history
                ])
            
            prompt = self.build_optimized_prompt(query, retrieved_texts, history_text)
            response = self.llm.invoke(prompt, query, book_name, retrieved_texts).content
            
            return response, query_embedding
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class OptimizedLlama3EduQuery:
    def __init__(self):
        self.app = FastAPI(title="Optimized Llama3 EduQuery", version="1.0.0")
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        chat_manager = OptimizedChatManager()
        pdf_processor = OptimizedPDFProcessor(collection)
        query_processor = OptimizedQueryProcessor(collection, chat_manager)
        embedder = OptimizedLlama3Embedder()

        @self.app.get("/")
        async def root():
            return {"message": "Optimized Llama3 EduQuery API is running", "status": "ok", "model": "llama3:8b"}

        @self.app.post("/upload/")
        async def upload_pdf(file: UploadFile = File(...)):
            try:
                if not file.filename.endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="Only PDF files are allowed")
                
                # Check if file already exists
                file_path = os.path.join(Config.PDF_DIR, file.filename)
                if os.path.exists(file_path):
                    # Check if embeddings already exist
                    try:
                        existing_results = collection.get(where={"source": file.filename})
                        if existing_results.get("ids"):
                            return JSONResponse(
                                status_code=200,
                                content={
                                    "message": "PDF already exists and is processed", 
                                    "filename": file.filename,
                                    "model": "llama3:8b",
                                    "status": "already_processed"
                                }
                            )
                    except Exception:
                        pass
                
                # Create temporary file for similarity check
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                try:
                    # Check for content similarity
                    is_duplicate, existing_filename = embedder.check_pdf_similarity(temp_file_path, collection)
                    
                    if is_duplicate:
                        os.unlink(temp_file_path)
                        raise HTTPException(
                            status_code=400, 
                            detail=f"A PDF with similar content already exists: {existing_filename}"
                        )
                    
                    # Save and process file
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)
                    
                    pdf_processor.process_pdf(file_path)
                    
                    return JSONResponse(
                        status_code=200, 
                        content={
                            "message": "PDF uploaded and processed successfully", 
                            "filename": file.filename,
                            "model": "llama3:8b"
                        }
                    )
                    
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Error uploading PDF: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {str(e)}")

        @self.app.get("/books/")
        async def list_books():
            try:
                books = [b for b in os.listdir(Config.PDF_DIR) if b.endswith('.pdf')]
                return {"books": books, "model": "llama3:8b"}
            except Exception as e:
                logging.error(f"Error listing books: {e}")
                return {"books": [], "error": str(e), "model": "llama3:8b"}

        @self.app.get("/book")
        async def get_pdf(subject: str = Query(...)):
            file_path = os.path.join(Config.PDF_DIR, subject)
            if os.path.isfile(file_path):
                return FileResponse(file_path, media_type='application/pdf', filename=subject, headers={"Content-Disposition": "inline"})
            return {"error": "PDF not found"}

        @self.app.post("/query/")
        async def query_pdf(query: str = Form(...), book_name: str = Form(...), 
                          session_id: str = Form(default="default"), debug: bool = Form(default=False)):
            try:
                book_specific_session = f"{session_id}_{book_name}"
                
                if is_casual_query(query):
                    casual_response = handle_casual_query(query)
                    chat_manager.add_to_history(book_specific_session, query, casual_response)
                    return {"answer": casual_response, "sources": [], "model": "llama3:8b"}
                
                chat_history = chat_manager.get_history(book_specific_session)
                query_to_use = rewrite_followup_query(query, chat_history)
                
                if is_followup_query(query):
                    logging.info(f"[Follow-up] Final Rewritten Query: '{query_to_use}' (original: '{query}')")
                else:
                    logging.info(f"[Query] Using as-is: '{query}'")
                
                documents, metadatas = query_processor.retrieve_relevant_content(query_to_use, book_name)
                
                if not documents:
                    return {"answer": "No relevant content found in the PDF for your query.", "sources": [], "model": "llama3:8b"}
                
                retrieved_texts = "\n\n".join(documents)
                response, query_embedding = query_processor.generate_response(query_to_use, retrieved_texts, book_specific_session, book_name)
                
                chat_manager.add_to_history(book_specific_session, query, response, query_embedding)
                
                result = {
                    "answer": response, 
                    "sources": [meta.get("source", book_name) for meta in metadatas],
                    "model": "llama3:8b"
                }
                
                if debug:
                    result["debug"] = {
                        "chunks_retrieved": len(documents),
                        "total_chunks_in_book": collection.count(),
                        "retrieved_content_preview": retrieved_texts[:1000] + "..." if len(retrieved_texts) > 1000 else retrieved_texts,
                        "book_specific_session": book_specific_session,
                        "cache_enabled": True,
                        "batch_processing": True,
                        "model_config": {
                            "temperature": Config.LLAMA_TEMPERATURE,
                            "top_p": Config.LLAMA_TOP_P,
                            "max_tokens": Config.LLAMA_MAX_TOKENS
                        }
                    }
                
                return result
                
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                return {"answer": f"Error processing query: {str(e)}", "sources": [], "model": "llama3:8b"}

        @self.app.post("/end_session/")
        async def end_session(session_id: str = Form(...), book_name: str = Form(...)):
            try:
                file_path = os.path.join(Config.PDF_DIR, book_name)
                removed = False
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed = True
                
                try:
                    collection.delete(where={"source": book_name})
                    logging.info(f"Successfully removed embeddings for {book_name}")
                except Exception as e:
                    logging.error(f"Failed to remove embeddings for {book_name}: {e}")
                
                if session_id in chat_manager.chat_histories:
                    del chat_manager.chat_histories[session_id]
                
                return {
                    "removed": removed, 
                    "filename": book_name, 
                    "message": "Session ended successfully",
                    "model": "llama3:8b"
                }
                
            except Exception as e:
                logging.error(f"Error ending session: {e}")
                return {"error": f"Failed to end session: {str(e)}", "model": "llama3:8b"}

# ============================================================================
# INITIALIZATION AND RUN
# ============================================================================

# Initialize configuration
Config.initialize()

# Setup logging with reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s [%(levelname)s] %(message)s',
    force=True
)

# Suppress specific loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Initialize ChromaDB
client = PersistentClient(path=Config.CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="pdf_collection")

# Create FastAPI application
app = OptimizedLlama3EduQuery().app

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=Config.FASTAPI_PORT) 