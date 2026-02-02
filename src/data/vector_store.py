"""
Vector Store for RAG
Document embedding and similarity search using sentence-transformers
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.document_processor import Chunk
from src.utils.logger import get_logger
from src.config.config import get_config

logger = get_logger("vector_store")

class VectorStore:
    """Vector store for semantic search"""
    
    def __init__(self):
        config = get_config()
        
        self.model_name = config['rag']['embedding_model']
        self.store_path = Path(config['rag']['vector_store_path'])
        self.top_k = config['rag']['top_k']
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Storage
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Load existing store if available
        self._load_store()
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to vector store"""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        new_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Add to store
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        logger.info(f"Vector store now contains {len(self.chunks)} chunks")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks or self.embeddings is None:
            logger.warning("Vector store is empty")
            return []
        
        top_k = top_k or self.top_k
        
        logger.debug(f"Searching for: {query}")
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'text': self.chunks[idx].text,
                    'metadata': self.chunks[idx].metadata
                })
        
        logger.debug(f"Found {len(results)} relevant chunks")
        return results
    
    def save_store(self):
        """Save vector store to disk"""
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)
            
            # Save chunks
            chunks_file = self.store_path / "chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save embeddings
            embeddings_file = self.store_path / "embeddings.npy"
            np.save(embeddings_file, self.embeddings)
            
            logger.info(f"Vector store saved to {self.store_path}")
            
        except Exception as e:
            logger.error("Error saving vector store", error=e)
    
    def _load_store(self):
        """Load vector store from disk"""
        try:
            chunks_file = self.store_path / "chunks.pkl"
            embeddings_file = self.store_path / "embeddings.npy"
            
            if chunks_file.exists() and embeddings_file.exists():
                logger.info("Loading existing vector store...")
                
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                self.embeddings = np.load(embeddings_file)
                
                logger.info(f"Loaded {len(self.chunks)} chunks from store")
            else:
                logger.info("No existing vector store found")
                
        except Exception as e:
            logger.error("Error loading vector store", error=e)
    
    def clear_store(self):
        """Clear all data from vector store"""
        self.chunks = []
        self.embeddings = None
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_chunks': len(self.chunks),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model': self.model_name
        }
