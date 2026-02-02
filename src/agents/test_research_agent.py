import sys
from pathlib import Path
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.research_agent import ResearchAgent
from src.utils.logger import get_logger

logger = get_logger("test_research_agent")

async def test_research_agent():
    """Test the complete research agent workflow"""
    try:
        print("\n" + "="*70)
        print("FINANCIAL RESEARCH AGENT - SYSTEM TEST")
        print("="*70)
        
        # Initialize agent
        logger.info("Initializing Research Agent...")
        agent = ResearchAgent()
        
        print("\n✅ Research Agent initialized successfully!")
        
        # Test queries
        test_queries = [
            "What is Tesla's current stock performance?",
            "What are Apple's main revenue streams?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print("\n" + "="*70)
            print(f"TEST {i}: {query}")
            print("="*70)
            
            logger.info(f"Testing query: {query}")
            
            # Conduct research
            result = await agent.research(query, use_web=True)
            
            # Display results
            print(f"\n📊 RESEARCH RESULTS:\n")
            print(f"Query: {result.query}")
            print(f"\n{'='*70}")
            print("ANSWER:")
            print(f"{'='*70}")
            print(result.answer)
            
            if result.sources:
                print(f"\n{'='*70}")
                print(f"SOURCES ({len(result.sources)}):")
                print(f"{'='*70}")
                for j, source in enumerate(result.sources, 1):
                    print(f"\n{j}. {source['title']}")
                    print(f"   URL: {source['url']}")
                    print(f"   Snippet: {source['snippet'][:100]}...")
            
            if result.context_used:
                print(f"\n{'='*70}")
                print(f"CONTEXT RETRIEVED: {len(result.context_used)} chunks")
                print(f"{'='*70}")
            
            print("\n✅ Research completed successfully!")
            
            # Wait between queries
            if i < len(test_queries):
                print("\n⏳ Waiting 3 seconds before next query...")
                await asyncio.sleep(3)
        
        # Display stats
        print("\n" + "="*70)
        print("AGENT STATISTICS")
        print("="*70)
        stats = agent.get_stats()
        print(f"Vector Store: {stats['vector_store']['total_chunks']} chunks stored")
        print(f"Embedding Model: {stats['vector_store']['model']}")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
        logger.info("Research agent test completed successfully!")
        
    except Exception as e:
        logger.error("Research agent test failed", error=e)
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_research_agent())
