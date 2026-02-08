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
        """Add chunks to vector store with batching for performance"""
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        try:
            # Limit total chunks for performance (max 800 chunks in store - reduced for memory)
            max_store_size = 800
            if len(self.chunks) + len(chunks) > max_store_size:
                excess = (len(self.chunks) + len(chunks)) - max_store_size
                logger.warning(f"Vector store limit reached. Removing {excess} oldest chunks")
                # Remove oldest chunks
                self.chunks = self.chunks[excess:]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[excess:]
            
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            
            # Extract texts
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings in batches for better performance
            logger.info("Generating embeddings...")
            batch_size = 32  # Process in batches
            new_embeddings_list = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size)
                new_embeddings_list.append(batch_embeddings)
            
            new_embeddings = np.vstack(new_embeddings_list) if len(new_embeddings_list) > 1 else new_embeddings_list[0]
            
            # Add to store
            self.chunks.extend(chunks)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
                
        except MemoryError as e:
            logger.error(f"MemoryError adding chunks: {e}. Clearing oldest chunks.")
            # Emergency cleanup - keep only last 400 chunks
            if len(self.chunks) > 400:
                self.chunks = self.chunks[-400:]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[-400:]
            raise
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
        
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
            # Bounds check to prevent index errors
            if idx >= len(self.chunks):
                logger.warning(f"Index {idx} out of bounds for chunks list (size: {len(self.chunks)})")
                continue
            
            # Lowered threshold from 0.1 to 0.05 for better recall
            if similarities[idx] > 0.05:
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(similarities[idx]),
                    'text': self.chunks[idx].text,
                    'metadata': self.chunks[idx].metadata
                })
        
        logger.debug(f"Found {len(results)} relevant chunks")
        return results
    
    def cleanup_old_chunks(self, keep_last_n: int = 500):
        """Cleanup old chunks to free memory"""
        try:
            if len(self.chunks) > keep_last_n:
                removed = len(self.chunks) - keep_last_n
                self.chunks = self.chunks[-keep_last_n:]
                if self.embeddings is not None:
                    self.embeddings = self.embeddings[-keep_last_n:]
                logger.info(f"Cleaned up {removed} old chunks, {keep_last_n} remaining")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get ALL chunks for a specific document
        
        Args:
            doc_id: Document ID to retrieve chunks for
            
        Returns:
            List of all chunks from the document with metadata
        """
        if not self.chunks:
            logger.warning("Vector store is empty")
            return []
        
        # Filter chunks by doc_id in metadata
        doc_chunks = []
        for idx, chunk in enumerate(self.chunks):
            if chunk.metadata.get('doc_id') == doc_id:
                doc_chunks.append({
                    'chunk': chunk,
                    'score': 1.0,  # All chunks are equally relevant for full doc
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'index': idx
                })
        
        logger.info(f"Retrieved {len(doc_chunks)} chunks for doc_id: {doc_id}")
        return doc_chunks
    
    def remove_document_by_id(self, doc_id: str) -> int:
        """Remove specific document without clearing entire store"""
        if not self.chunks:
            logger.warning("Vector store is empty, nothing to remove")
            return 0
        
        # Find indices of chunks to remove
        indices_to_remove = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.metadata.get('doc_id') == doc_id
        ]
        
        if not indices_to_remove:
            logger.warning(f"No chunks found for doc_id: {doc_id}")
            return 0
        
        # Remove chunks and embeddings
        self.chunks = [
            chunk for i, chunk in enumerate(self.chunks)
            if i not in indices_to_remove
        ]
        
        if self.embeddings is not None:
            mask = np.ones(len(self.embeddings), dtype=bool)
            mask[indices_to_remove] = False
            self.embeddings = self.embeddings[mask] if self.chunks else None
        
        removed_count = len(indices_to_remove)
        logger.info(f"Removed {removed_count} chunks for doc_id: {doc_id}, {len(self.chunks)} chunks remaining")
        return removed_count
    
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
