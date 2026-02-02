"""
Document Processing
Text chunking and preprocessing for RAG
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re
from src.utils.logger import get_logger

logger = get_logger("document_processor")

@dataclass
class Document:
    """Document with metadata"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str = None

@dataclass
class Chunk:
    """Text chunk with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str = None

class DocumentProcessor:
    """Process and chunk documents for RAG"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document processor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)\[\]\"\']+', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of text chunks with metadata
        """
        if not text:
            return []
        
        # Clean text
        text = self.clean_text(text)
        
        chunks = []
        start = 0
        chunk_num = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    **(metadata or {}),
                    'chunk_num': chunk_num,
                    'start_pos': start,
                    'end_pos': end
                }
                
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=f"chunk_{chunk_num}"
                ))
                
                chunk_num += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0:
                start = end
        
        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def process_document(self, document: Document) -> List[Chunk]:
        """Process a document into chunks"""
        logger.info(f"Processing document: {document.metadata.get('title', 'Unknown')}")
        
        chunks = self.chunk_text(document.content, document.metadata)
        
        logger.info(f"Document processed into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        """Process multiple documents"""
        logger.info(f"Processing {len(documents)} documents")
        
        all_chunks = []
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
