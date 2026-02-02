import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.config import get_config

# Test the config
config = get_config()
print("✅ Configuration loaded successfully!")
print(f"App: {config.get('app.name')} v{config.get('app.version')}")
print(f"Environment: {config.get('app.environment')}")
print(f"OpenAI Model: {config.get('apis.openai.model')}")
print(f"Logging Level: {config.get('logging.level')}")
print(f"RAG Embedding: {config.get('rag.embedding_model')}")
