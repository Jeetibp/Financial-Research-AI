"""
Web Scraping Utilities
Extract content from web pages
"""

import httpx
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("web_scraper")

class WebScraper:
    """Web scraping utility"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch raw HTML from URL"""
        try:
            logger.debug(f"Fetching page: {url}")
            
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                
                logger.debug(f"Page fetched successfully: {len(response.text)} chars")
                return response.text
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}", error=e)
            return None
    
    async def extract_text(self, url: str) -> Optional[str]:
        """Extract clean text from webpage"""
        html = await self.fetch_page(url)
        if not html:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            logger.debug(f"Extracted {len(text)} characters from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}", error=e)
            return None
    
    async def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from webpage"""
        html = await self.fetch_page(url)
        if not html:
            return {}
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            metadata = {
                'url': url,
                'title': None,
                'description': None,
                'keywords': None
            }
            
            # Title
            if soup.title:
                metadata['title'] = soup.title.string
            
            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property_attr = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description' or property_attr == 'og:description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}", error=e)
            return {}
