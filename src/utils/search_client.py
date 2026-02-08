"""
Search Engine Client
Web search using DuckDuckGo (free, no API key needed)
"""

import httpx
from typing import List, Dict, Any
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

logger = get_logger("search_client")

class SearchClient:
    """DuckDuckGo search client (no API key required)"""
    
    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo with validation and error handling"""
        try:
            # Input validation
            if not query or not isinstance(query, str):
                logger.warning("Invalid search query provided")
                return []
            
            query = query.strip()
            if len(query) < 2:
                logger.warning("Search query too short")
                return []
            
            # Limit max results to prevent memory issues
            max_results = min(max_results, 20)
            
            logger.info(f"Searching for: {query}")
            
            # Prepare search
            params = {
                'q': query,
                'kl': 'us-en'
            }
            
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.post(
                    self.base_url,
                    data=params,
                    headers=self.headers
                )
                response.raise_for_status()
                
                # Limit response size
                html = response.text
                max_html_size = 5_000_000  # 5MB limit
                if len(html) > max_html_size:
                    logger.warning(f"Search response too large, truncating")
                    html = html[:max_html_size]
                
                # Parse results
                results = self._parse_results(html, max_results)
                
                logger.info(f"Found {len(results)} results for: {query}")
                return results
                
        except httpx.TimeoutException:
            logger.error(f"Search timeout for query '{query}'")
            return []
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during search for '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Search error for query '{query}'", error=e)
            return []
    
    def _parse_results(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse search results from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Find all result divs
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:max_results]:
                try:
                    # Title and URL
                    title_tag = div.find('a', class_='result__a')
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    url = title_tag.get('href', '')
                    
                    # Snippet
                    snippet_tag = div.find('a', class_='result__snippet')
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error("Error parsing search results", error=e)
            return []
    
    async def search_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for news articles"""
        news_query = f"{query} news"
        return await self.search(news_query, max_results)
    
    async def search_financial(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for financial information"""
        financial_query = f"{query} financial report analysis"
        return await self.search(financial_query, max_results)
