import sys
from pathlib import Path
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.search_client import SearchClient
from src.utils.web_scraper import WebScraper
from src.utils.logger import get_logger

logger = get_logger("test_search")

async def test_search():
    """Test search functionality"""
    try:
        logger.info("Testing Search Client...")
        
        client = SearchClient()
        
        # Test search
        results = await client.search("Tesla stock price", max_results=3)
        
        print("\n" + "="*60)
        print("SEARCH RESULTS:")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
        
        print("\n" + "="*60)
        logger.info(f"✅ Search test successful! Found {len(results)} results")
        
        # Test scraper
        if results:
            logger.info("Testing Web Scraper...")
            scraper = WebScraper()
            
            first_url = results[0]['url']
            text = await scraper.extract_text(first_url)
            
            if text:
                print(f"\n📄 Scraped {len(text)} characters from first result")
                print(f"Preview: {text[:200]}...")
                logger.info("✅ Web scraper test successful!")
            else:
                print("⚠️ Could not scrape content")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error("Test failed", error=e)
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_search())
