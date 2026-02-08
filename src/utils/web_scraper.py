"""
Web Scraping Utilities
Extract content from web pages
"""

import httpx
import asyncio
import random
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("web_scraper")

class WebScraper:
    """Web scraping utility with retry and anti-blocking"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
    
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch raw HTML from URL with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching page (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                # Rotate user agent
                headers = {
                    'User-Agent': random.choice(self.user_agents),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Referer': 'https://www.google.com/'
                }
                
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    
                    logger.debug(f"Page fetched successfully: {len(response.text)} chars")
                    return response.text
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning(f"403 Forbidden on attempt {attempt + 1}/{self.max_retries} for {url}")
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Waiting {wait_time:.1f}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Error fetching {url}", error=e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None
        
        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None
    
    async def extract_text(self, url: str) -> Optional[str]:
        """Extract clean text from webpage"""
        html = await self.fetch_page(url)
        if not html:
            return None
        
        # Limit HTML size to prevent memory issues (10MB limit)
        max_html_size = 10_000_000
        if len(html) > max_html_size:
            logger.warning(f"HTML too large ({len(html):,} bytes), truncating to {max_html_size:,} bytes")
            html = html[:max_html_size]
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Limit extracted text size (2MB limit)
            max_text_size = 2_000_000
            if len(text) > max_text_size:
                logger.warning(f"Extracted text too large ({len(text):,} chars), truncating to {max_text_size:,} chars")
                text = text[:max_text_size]
            
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
