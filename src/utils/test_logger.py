import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

# Test the logger
logger = get_logger("test")

logger.info("✅ Logger initialized successfully")
logger.debug("This is a debug message")
logger.warning("⚠️ This is a warning")
logger.error("❌ This is an error test")

print("\n✅ Logger test complete! Check logs/agent.log file!")
