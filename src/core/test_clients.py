import sys
from pathlib import Path
import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger("test_clients")

async def test_llm_client():
    """Test OpenAI LLM client"""
    try:
        logger.info("Testing LLM Client...")
        
        client = LLMClient()
        
        # Simple test query
        response = await client.generate_with_system_prompt(
            system_prompt="You are a helpful assistant. Respond in one sentence.",
            user_message="What is 2+2?"
        )
        
        print(f"\n✅ LLM Response: {response}\n")
        logger.info("LLM Client test successful!")
        
    except Exception as e:
        logger.error("LLM Client test failed", error=e)
        print(f"\n❌ Test failed: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(test_llm_client())
