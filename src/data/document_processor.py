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
        self.logger = get_logger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Increase chunk size to reduce total chunks (better performance)
        if chunk_size < 1500:
            self.chunk_size = 1500
            self.logger.info(f"Increased chunk_size to {self.chunk_size} for better performance")
        self.logger.info(f"Document processor initialized: chunk_size={self.chunk_size}, overlap={chunk_overlap}")
    
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
        
        # Safety check - limit text size before processing
        max_text_length = 1_500_000  # 1.5MB character limit
        if len(text) > max_text_length:
            self.logger.warning(f"Text too large ({len(text):,} chars), truncating to {max_text_length:,}")
            text = text[:max_text_length]
        
        # Clean text
        text = self.clean_text(text)
        
        chunks = []
        start = 0
        chunk_num = 0
        max_chunks = 500  # Reduced from 2000 for better performance
        
        while start < len(text) and chunk_num < max_chunks:
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
                try:
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
                except MemoryError:
                    self.logger.error(f"MemoryError at chunk {chunk_num}, stopping chunking")
                    break
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0:
                start = end
        
        if chunk_num >= max_chunks:
            self.logger.warning(f"Reached maximum chunk limit ({max_chunks})")
        
        self.logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def process_document(self, document: Document) -> List[Chunk]:
        """Process a single document into chunks"""
        try:
            # SAFE ACCESS - use getattr with fallback
            doc_title = getattr(document, 'title', getattr(document, 'url', 'Unknown'))
            self.logger.info(f"Processing document: {doc_title}")
            
            # CHECK AND LIMIT SIZE - reduced to 1MB for safety
            content_size = len(document.content)
            max_size = 1_000_000  # 1MB limit (reduced from 3MB)
            
            if content_size > max_size:
                self.logger.warning(f"Document too large ({content_size:,} bytes > {max_size:,}), truncating")
                document.content = document.content[:max_size]
            
            # Clean content and compress whitespace
            document.content = ' '.join(document.content.split())
            
            chunks = self.chunk_text(document.content, document.metadata)
            self.logger.info(f"Document processed into {len(chunks)} chunks")
            return chunks
            
        except MemoryError as e:
            doc_title = getattr(document, 'title', getattr(document, 'url', 'Unknown'))
            self.logger.error(f"MemoryError processing {doc_title} - Skipping document")
            return []
        except Exception as e:
            doc_title = getattr(document, 'title', getattr(document, 'url', 'Unknown'))
            self.logger.error(f"Error processing document {doc_title}: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        """Process multiple documents into chunks"""
        self.logger.info(f"Processing {len(documents)} documents")
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                # SAFE ACCESS
                doc_title = getattr(doc, 'title', getattr(doc, 'url', 'Unknown'))
                self.logger.error(f"Failed to process {doc_title}: {e}")
                continue
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
