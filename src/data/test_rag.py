import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.document_processor import Document, DocumentProcessor
from src.data.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger("test_rag")

def test_rag_system():
    """Test RAG document processing and retrieval"""
    try:
        logger.info("Testing RAG System...")
        
        # Sample financial documents
        documents = [
            Document(
                content="""
                Tesla Inc reported strong Q4 2023 earnings with revenue of $25.2 billion,
                beating analyst expectations. The company's automotive segment showed 
                significant growth with increased vehicle deliveries. Net income reached 
                $7.9 billion for the quarter. CEO Elon Musk highlighted improvements in 
                manufacturing efficiency and cost reduction initiatives.
                """,
                metadata={'source': 'Tesla Q4 Report', 'date': '2023-Q4', 'company': 'Tesla'}
            ),
            Document(
                content="""
                Apple Inc announced record-breaking services revenue of $85 billion in 2023.
                The iPhone continues to dominate smartphone market share. Apple's investment
                in AI and machine learning capabilities is expected to drive future growth.
                The company maintains strong cash reserves exceeding $160 billion.
                """,
                metadata={'source': 'Apple Annual Report', 'date': '2023', 'company': 'Apple'}
            ),
            Document(
                content="""
                Microsoft Azure cloud services grew 30% year-over-year, contributing significantly
                to Microsoft's revenue growth. The integration of OpenAI technologies into Microsoft
                products has been well-received. Office 365 subscriptions continue to increase,
                and the gaming division showed strong performance with Xbox and game pass services.
                """,
                metadata={'source': 'Microsoft Quarterly Report', 'date': '2023-Q3', 'company': 'Microsoft'}
            )
        ]
        
        print("\n" + "="*60)
        print("STEP 1: Processing Documents")
        print("="*60)
        
        # Process documents
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        all_chunks = processor.process_documents(documents)
        
        print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        print("\n" + "="*60)
        print("STEP 2: Building Vector Store")
        print("="*60)
        
        # Build vector store
        vector_store = VectorStore()
        vector_store.add_chunks(all_chunks)
        
        stats = vector_store.get_stats()
        print(f"✅ Vector store built:")
        print(f"   - Total chunks: {stats['total_chunks']}")
        print(f"   - Embedding dimension: {stats['embedding_dim']}")
        print(f"   - Model: {stats['model']}")
        
        print("\n" + "="*60)
        print("STEP 3: Testing Search Queries")
        print("="*60)
        
        # Test queries
        queries = [
            "What were Tesla's earnings?",
            "Tell me about Apple's revenue",
            "Microsoft cloud services performance"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            results = vector_store.search(query, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i} (Score: {result['score']:.3f}):")
                print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"   Text: {result['text'][:150]}...")
        
        print("\n" + "="*60)
        print("✅ RAG System Test Complete!")
        print("="*60)
        
        logger.info("RAG system test successful!")
        
    except Exception as e:
        logger.error("RAG system test failed", error=e)
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_rag_system()
