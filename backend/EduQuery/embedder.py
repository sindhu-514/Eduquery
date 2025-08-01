import logging
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from EduQuery.config import Config
from typing import List
import time
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class OllamaEmbedder:
    def __init__(self, model_name=Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        self.session = requests.Session()
        self.session.timeout = 30
    
    def check_pdf_similarity(self, uploaded_file_path, collection, similarity_threshold=0.95):
        """
        Check if a PDF with similar content already exists in the database.
        Returns (is_duplicate, existing_filename) if duplicate found, (False, None) otherwise.
        """
        try:
            # Extract first few pages from uploaded PDF
            loader = PyPDFLoader(uploaded_file_path)
            pages = loader.load()
            
            # Use first 3 pages or all pages if less than 3
            sample_pages = pages[:min(3, len(pages))]
            sample_texts = [page.page_content.strip() for page in sample_pages if page.page_content.strip()]
            
            if not sample_texts:
                return False, None
            
            # Split sample texts into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            documents = [page for page in sample_texts]
            chunks = text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]
            
            # Generate embeddings for sample chunks
            sample_embeddings = self.embed_documents_batch(chunk_texts)
            
            # Get all existing embeddings from the collection
            existing_results = collection.get()
            if not existing_results['embeddings']:
                return False, None
            
            existing_embeddings = existing_results['embeddings']
            existing_metadatas = existing_results['metadatas']
            
            # Group embeddings by source file
            source_groups = {}
            for i, metadata in enumerate(existing_metadatas):
                source = metadata.get('source', 'unknown')
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(existing_embeddings[i])
            
            # Check similarity with each existing PDF
            for source_name, source_embeddings in source_groups.items():
                max_similarity = 0
                
                # Compare each sample embedding with embeddings from this source
                for sample_emb in sample_embeddings:
                    for existing_emb in source_embeddings:
                        # Calculate cosine similarity
                        similarity = np.dot(sample_emb, existing_emb) / (np.linalg.norm(sample_emb) * np.linalg.norm(existing_emb))
                        max_similarity = max(max_similarity, similarity)
                
                # If any sample has high similarity with this source, consider it a duplicate
                if max_similarity >= similarity_threshold:
                    logging.info(f"PDF similarity check: {uploaded_file_path} is {max_similarity:.3f} similar to existing {source_name}")
                    return True, source_name
            
            return False, None
            
        except Exception as e:
            logging.error(f"Error checking PDF similarity: {e}")
            return False, None

    def embed_query(self, text: str):
        """Generate embedding for a single text query"""
        try:
            response = self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}")
            raise
    
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Production-ready batch embedding with multiple strategies:
        1. Try concurrent individual embeddings (fastest for Ollama)
        2. Fallback to sequential if concurrent fails
        3. Use zero embeddings for failed texts
        """
        if not texts:
            return []
        
        logging.info(f"Starting production batch embedding for {len(texts)} documents.")
        start_time = time.time()
        
        # Strategy 1: Concurrent individual embeddings (recommended for Ollama)
        try:
            embeddings = self._embed_concurrent(texts)
            elapsed = time.time() - start_time
            logging.info(f"Concurrent embedding completed: {len(embeddings)} embeddings in {elapsed:.2f}s")
            return embeddings
        except Exception as e:
            logging.warning(f"Concurrent embedding failed: {e}, falling back to sequential")
            
        # Strategy 2: Sequential individual embeddings
        try:
            embeddings = self._embed_sequential(texts)
            elapsed = time.time() - start_time
            logging.info(f"Sequential embedding completed: {len(embeddings)} embeddings in {elapsed:.2f}s")
            return embeddings
        except Exception as e:
            logging.error(f"Sequential embedding failed: {e}, using zero embeddings")
            
        # Strategy 3: Zero embeddings fallback
        return self._create_zero_embeddings(texts)

    def _embed_concurrent(self, texts: List[str]) -> List[List[float]]:
        """Use ThreadPoolExecutor for concurrent individual embeddings"""
        embeddings = [None] * len(texts)
        max_workers = min(Config.MAX_WORKERS, len(texts), 10)  # Limit concurrent requests
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all embedding tasks
            future_to_index = {
                executor.submit(self._embed_single_with_retry, text): i 
                for i, text in enumerate(texts)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[index] = embedding
                    completed += 1
                    
                    # Log progress every 10 embeddings
                    if completed % 10 == 0:
                        logging.info(f"Concurrent embedding progress: {completed}/{len(texts)}")
                        
                except Exception as e:
                    logging.error(f"Failed to embed text {index}: {e}")
                    embeddings[index] = self._get_zero_embedding()
                    completed += 1
        
        # Verify all embeddings are complete
        if None in embeddings:
            logging.error("Some embeddings failed, replacing with zero embeddings")
            embeddings = [emb if emb is not None else self._get_zero_embedding() for emb in embeddings]
        
        return embeddings

    def _embed_sequential(self, texts: List[str]) -> List[List[float]]:
        """Sequential individual embeddings with progress tracking"""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self._embed_single_with_retry(text)
                embeddings.append(embedding)
                
                # Log progress every 10 embeddings
                if (i + 1) % 10 == 0:
                    logging.info(f"Sequential embedding progress: {i + 1}/{len(texts)}")
                    
            except Exception as e:
                logging.error(f"Failed to embed text {i}: {e}")
                embeddings.append(self._get_zero_embedding())
        
        return embeddings

    def _embed_single_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """Embed single text with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.embed_query(text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Embedding attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    def _get_zero_embedding(self) -> List[float]:
        """Get zero embedding with correct dimension for the model"""
        # bge-m3 uses 1024 dimensions
        return [0.0] * 1024

    def _create_zero_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create zero embeddings for all texts (emergency fallback)"""
        logging.warning(f"Creating zero embeddings for {len(texts)} texts")
        return [self._get_zero_embedding() for _ in texts]

    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()