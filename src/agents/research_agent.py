"""
Research Agent - Advanced Version
Orchestrates web search, scraping, RAG, and LLM generation with smart query handling
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from src.core.llm_client import LLMClient
from src.utils.search_client import SearchClient
from src.utils.web_scraper import WebScraper
from src.data.document_processor import Document, DocumentProcessor
from src.data.vector_store import VectorStore
from src.utils.logger import get_logger
from src.config.config import get_config
from src.utils.financial_calculator import FinancialCalculator
from src.utils.smart_calculator import SmartFinancialCalculator

logger = get_logger("research_agent")

@dataclass
class ResearchResult:
    """Research result with sources and research reasoning"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: List[str]
    research_reasoning: Optional[List[Dict[str, Any]]] = None
    iteration_count: int = 1
    
    def format_with_reasoning(self) -> str:
        """Format report with research progression visible"""
        report = f"# Research Report: {self.query}\n\n"
        
        if self.research_reasoning and len(self.research_reasoning) > 1:
            report += "## 🔬 Research Process\n\n"
            report += f"*Conducted {self.iteration_count} iterations of deep research*\n\n"
            for i, step in enumerate(self.research_reasoning, 1):
                report += f"**Step {i}**: {step['action']}\n"
                report += f"*Reasoning*: {step['reasoning']}\n\n"
        
        report += f"## 📊 Analysis\n\n{self.answer}\n\n"
        
        if self.sources:
            # Deduplicate sources by URL
            seen_urls = set()
            unique_sources = []
            for source in self.sources:
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            
            report += f"## 📚 Sources ({len(unique_sources)})\n\n"
            for i, source in enumerate(unique_sources, 1):
                title = source.get('title', 'Untitled')
                url = source.get('url', '#')
                snippet = source.get('snippet', '')
                
                report += f"{i}. **[{title}]({url})**\n"
                if snippet:
                    report += f"   *{snippet[:150]}...*\n\n"
                else:
                    report += "\n"
        
        return report

class ResearchAgent:
    """AI Research Agent for financial analysis with smart query handling"""
    
    def __init__(self):
        logger.info("Initializing Research Agent...")
        
        self.config = get_config()
        self.llm = LLMClient()
        self.search_client = SearchClient()
        self.scraper = WebScraper()
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config['rag']['chunk_size'],
            chunk_overlap=self.config['rag']['chunk_overlap']
        )
        self.vector_store = VectorStore()
        self.calculator = FinancialCalculator()  # Python calculator
        self.smart_calc = SmartFinancialCalculator()  # Smart year/calculation detection
        
        # Predefined responses for common queries
        self.quick_responses = {
            'greetings': [
                'hi', 'hello', 'hey', 'good morning', 'good evening', 
                'good afternoon', 'greetings'
            ],
            'thanks': ['thanks', 'thank you', 'thx', 'appreciate'],
            'help': ['help', 'what can you do', 'how do you work', 'capabilities']
        }
        
        logger.info("Research Agent initialized successfully")
    
    def _is_financial_query(self, query: str) -> bool:
        """Determine if query is financial/requires research"""
        financial_keywords = [
            'stock', 'price', 'market', 'company', 'revenue', 'earnings',
            'financial', 'invest', 'analysis', 'performance', 'quarter',
            'profit', 'loss', 'share', 'dividend', 'valuation', 'growth',
            'sector', 'industry', 'economy', 'trend', 'forecast', 'report'
        ]
        
        query_lower = query.lower()
        
        # Check for financial keywords
        has_financial_keyword = any(keyword in query_lower for keyword in financial_keywords)
        
        # Check if query is long enough (likely a real question)
        is_substantial = len(query.split()) >= 3
        
        return has_financial_keyword or is_substantial
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for quick responses"""
        query_lower = query.lower().strip()
        
        # Check greetings
        if any(greeting in query_lower for greeting in self.quick_responses['greetings']):
            return 'greeting'
        
        # Check thanks
        if any(thanks in query_lower for thanks in self.quick_responses['thanks']):
            return 'thanks'
        
        # Check help
        if any(help_word in query_lower for help_word in self.quick_responses['help']):
            return 'help'
        
        return 'research'
    
    def _get_quick_response(self, query_type: str) -> str:
        """Get predefined quick response"""
        responses = {
            'greeting': """Hello! 👋 I'm your **Financial Research AI** assistant.

I specialize in providing real-time financial analysis and market research. Here's what I can help you with:

📈 **Stock Analysis**
• Real-time stock prices and performance metrics
• Technical and fundamental analysis
• Price trends and forecasts

💼 **Company Research**
• Financial statements and earnings reports
• Revenue streams and business models
• Competitive analysis

📊 **Market Intelligence**
• Sector trends and market dynamics
• Economic indicators and news
• Investment opportunities

💡 **Try asking:**
• "What is Tesla's current stock performance?"
• "Analyze Apple's Q4 earnings"
• "Compare Amazon and Google revenue growth"
• "What are the AI sector market trends?"

What would you like to research today?""",
            
            'thanks': """You're very welcome! 😊 

I'm here whenever you need financial insights or market research. Feel free to ask me anything about stocks, companies, markets, or investment opportunities.

Is there anything else you'd like to explore?""",
            
            'help': """🤖 **How I Work:**

I'm an AI-powered financial research assistant that:

1️⃣ **Searches the Web** - I find the latest financial data from reliable sources
2️⃣ **Analyzes Content** - I extract and process relevant information
3️⃣ **Generates Insights** - I provide comprehensive analysis using AI
4️⃣ **Cites Sources** - All information includes verifiable sources

💪 **My Capabilities:**
• Real-time stock market data
• Company financial analysis
• Market trends and forecasts
• Economic indicators
• Sector comparisons
• Investment research

⚡ **Best Results:**
Ask specific questions like:
• "What is [company]'s stock price?"
• "Analyze [company]'s latest earnings"
• "Compare [company A] vs [company B]"
• "What are trends in [sector/industry]?"

Ready to start? Ask me anything about finance!"""
        }
        
        return responses.get(query_type, "")
    
    async def research(self, query: str, use_web: bool = True, company_info: Optional[Dict] = None, deep_mode: bool = False) -> ResearchResult:
        """
        Conduct research on a query with smart handling
        
        Args:
            query: Research question
            use_web: Whether to search web for new information
            company_info: Resolved company information from Universal Company Resolver
            deep_mode: Enable deep iterative research mode (5-15 iterations)
            
        Returns:
            ResearchResult with answer and sources
        """
        logger.info(f"Starting research for query: {query}")
        
        # Route to deep research mode if requested
        if deep_mode:
            logger.info("🔬 DEEP RESEARCH MODE activated")
            return await self.deep_research_mode(query, company_info=company_info, max_iterations=15)
        
        # Log company info if provided
        if company_info and company_info.get('resolved'):
            if company_info.get('multiple'):
                companies = ', '.join([c['company'] for c in company_info['companies']])
                logger.info(f"[Company Info] Multiple companies: {companies}")
            else:
                logger.info(f"[Company Info] {company_info.get('company')} ({company_info.get('ticker')})")
        
        # Improved greeting detection - only treat as greeting if:
        # 1. Query is ONLY a greeting (exact match) OR
        # 2. Query has greeting keyword AND is short (< 5 words)
        # 3. AND query doesn't contain financial keywords
        
        greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "how are you"]
        financial_keywords = ["stock", "revenue", "profit", "price", "company", "analysis", "market", 
                             "compare", "microsoft", "google", "apple", "tesla", "amazon", "which", 
                             "earnings", "financial", "sector", "industry"]
        
        query_lower = query.lower().strip()
        
        # Check if it's only a greeting
        is_greeting = (
            any(query_lower == keyword for keyword in greeting_keywords) or
            (any(keyword in query_lower for keyword in greeting_keywords) and len(query.split()) < 5)
        )
        
        # Check for financial content
        has_financial_content = any(keyword in query_lower for keyword in financial_keywords)
        
        # Only return greeting if it's a greeting AND no financial content
        if is_greeting and not has_financial_content:
            logger.info("Quick response for greeting")
            return ResearchResult(
                answer=self._get_quick_response('greeting'),
                sources=[],
                query=query,
                context_used=[]
            )
        
        # Check for thanks
        thanks_keywords = ['thanks', 'thank you', 'thx', 'appreciate']
        if any(query_lower == keyword for keyword in thanks_keywords):
            logger.info("Quick response for thanks")
            return ResearchResult(
                answer=self._get_quick_response('thanks'),
                sources=[],
                query=query,
                context_used=[]
            )
        
        # Check for help (only if exact match or very short)
        help_keywords = ['help', 'what can you do', 'how do you work']
        if any(query_lower == keyword for keyword in help_keywords) and not has_financial_content:
            logger.info("Quick response for help")
            return ResearchResult(
                answer=self._get_quick_response('help'),
                sources=[],
                query=query,
                context_used=[]
            )
        
        # All other queries proceed with full research
        logger.info("Proceeding with full research pipeline")
        
        # CHECK FOR BATCH/MULTIPLE CALCULATIONS FIRST
        if self._is_batch_query(query):
            logger.info("🔢 Batch calculation request detected")
            batch_result = await self._handle_batch_calculations(query)
            if batch_result:
                return batch_result
        
        # PRIORITY FIX: Check for simple calculations FIRST (before web search)
        # If query has numeric values and calculation keywords, try calculator immediately
        if self._has_inline_values(query):
            logger.info("🧮 Detected calculation with inline values - attempting direct calculation")
            calculation_result = await self._calculate_with_inline_values(query)
            if calculation_result:
                logger.info("✅ Calculation successful - returning result without web search")
                return ResearchResult(
                    answer=calculation_result,
                    sources=[],
                    query=query,
                    context_used=["Financial Calculator (Direct Computation)"]
                )
        
        # TYPE 2 FIX: Check for calculation queries WITHOUT inline values (need to fetch data)
        # Examples: "Calculate P/E for TCS", "What is Reliance ROE?", "Compare TCS and Infosys P/E"
        if self._is_calculation_query(query) and not self._has_inline_values(query):
            logger.info("🔍 Calculation query detected - checking if data fetch is needed")
            
            # If company_info not provided, try to extract from query
            if not company_info or not company_info.get('resolved'):
                logger.info("Company not resolved by router - attempting to extract from query")
                company_info = await self._extract_company_from_query(query)
            
            # Check if company is mentioned/resolved
            if company_info and company_info.get('resolved'):
                logger.info("📊 Company resolved - attempting fetch and calculate")
                fetch_calc_result = await self._handle_fetch_and_calculate(query, company_info)
                
                if fetch_calc_result:
                    logger.info("✅ Fetch & Calculate successful")
                    return fetch_calc_result
                else:
                    logger.info("⚠️ Could not fetch required data - falling back to research")
        
        # Initialize calculation_result for later attempts
        calculation_result = None
        
        # Detect if this is a stock-related query (expanded keywords)
        stock_price_keywords = ['stock price', 'stock today', 'share price', 'current price', 'opening price', 
                                'today opened', 'market open', 'current stock', 'price today']
        stock_analysis_keywords = ['pharma stock', 'pharmaceutical stock', 'top stock', 'best stock', 
                                  'stock performance', 'stock growth', 'stock analysis', 'stock that',
                                  'stock valuation', 'stock market', 'shares', 'equity', 'ticker']
        
        is_stock_price_query = any(keyword in query_lower for keyword in stock_price_keywords)
        is_stock_analysis = any(keyword in query_lower for keyword in stock_analysis_keywords)
        
        # Extract company/ticker symbols from query OR use company_info
        stock_data_context = ""
        if is_stock_price_query or is_stock_analysis or (company_info and company_info.get('resolved')):
            if company_info and company_info.get('resolved'):
                # Use validated company info from resolver
                stock_data_context = await self._fetch_stock_prices_from_company_info(company_info)
                logger.info(f"Fetched stock data from company_info: {len(stock_data_context)} chars")
            else:
                # Fallback: Try to extract companies from query text
                logger.info("Company not resolved, attempting to extract from query...")
                extracted_companies = await self._extract_companies_from_text(query)
                if extracted_companies:
                    logger.info(f"Extracted {len(extracted_companies)} companies from query")
                    stock_data_context = await self._fetch_multiple_stock_prices(extracted_companies)
                    logger.info(f"Fetched stock price data: {len(stock_data_context)} chars")
                else:
                    # Last resort: Old method
                    stock_data_context = await self._fetch_stock_prices_from_query(query)
                    logger.info(f"Fetched stock price data (fallback): {len(stock_data_context)} chars")
        
        # Proceed with full research
        sources = []
        context_chunks = []
        
        # Step 1: Search web if requested (PARALLEL PROCESSING)
        if use_web:
            logger.info("Searching web for information...")
            # INCREASED from 3 to 6 sources for better coverage
            search_results = await self.search_client.search(query, max_results=6)
            
            if search_results:
                # Scrape top results in parallel (scrape up to 3 pages)
                scraped_docs = await self._scrape_results_parallel(search_results[:3])
                
                if scraped_docs:
                    # Process and add to vector store
                    chunks = self.doc_processor.process_documents(scraped_docs)
                    self.vector_store.add_chunks(chunks)
                    
                    # Return 3-6 sources instead of always 3
                    sources.extend([{
                        'title': sr['title'],
                        'url': sr['url'],
                        'snippet': sr['snippet']
                    } for sr in search_results[:6]])
        
        # Step 2: Retrieve relevant context from vector store
        logger.info("Retrieving relevant context from vector store...")
        retrieved_chunks = self.vector_store.search(query, top_k=5)
        
        # Initialize context_chunks list
        context_chunks = []
        if retrieved_chunks:
            context_chunks = [chunk['text'] for chunk in retrieved_chunks]
            logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
        else:
            logger.info("No chunks retrieved from vector store")
        
        # Add stock price data to context if available
        if stock_data_context:
            context_chunks.insert(0, stock_data_context)  # Add at beginning for priority
            logger.info("Added stock price data to context")
        
        # Try calculation again with context (now we have data from web/vector store)
        if not calculation_result:
            calculation_result = await self._try_calculate(query, context_chunks)
        
        # If still no calculation data, try targeted financial search
        if not calculation_result and self._is_calculation_query(query):
            logger.info("Calculation detected but no data - performing targeted financial search")
            financial_context = await self._search_for_financial_data(query, company_info)
            if financial_context:
                context_chunks.insert(0, financial_context)
                calculation_result = await self._try_calculate(query, context_chunks)
        
        # Step 3: Generate answer using LLM or return calculation if available
        if calculation_result:
            logger.info("Using calculator result as primary answer")
            # Add sources info if we found them through search
            if sources:
                sources_text = "\n\n### 📚 Sources\n\n" + "\n".join([
                    f"- [{s['title']}]({s['url']})" for s in sources[:3]
                ])
                answer = calculation_result + sources_text
            else:
                answer = calculation_result
        else:
            logger.info("Generating answer with LLM...")
            answer = await self._generate_answer(query, context_chunks)
        
        result = ResearchResult(
            answer=answer,
            sources=sources,
            query=query,
            context_used=context_chunks
        )
        
        logger.info("Research completed successfully")
        return result
    
    async def _scrape_results_parallel(self, search_results: List[Dict[str, Any]]) -> List[Document]:
        """Scrape content from search results in parallel (FASTER)"""
        documents = []
        
        # Create all scraping tasks
        scraping_tasks = [
            self.scraper.extract_text(result['url'])
            for result in search_results
        ]
        
        # Execute all tasks in parallel
        scraped_texts = await asyncio.gather(*scraping_tasks, return_exceptions=True)
        
        # Process results
        for i, text in enumerate(scraped_texts):
            if isinstance(text, Exception):
                logger.debug(f"Failed to scrape {search_results[i]['url']}: {text}")
                continue
            
            if text and len(text) > 100:  # Minimum content length
                documents.append(Document(
                    content=text[:2000],  # Limit to 2000 chars for speed
                    metadata={
                        'title': search_results[i]['title'],
                        'url': search_results[i]['url'],
                        'source': 'web_search'
                    }
                ))
                logger.debug(f"Scraped: {search_results[i]['title']}")
        
        logger.info(f"Successfully scraped {len(documents)} documents")
        return documents
    
    async def _generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using LLM with context"""
        
        # Build context
        if context_chunks:
            context = "\n\n".join([
                f"[Source {i+1}]\n{chunk}"
                for i, chunk in enumerate(context_chunks[:5])
            ])
        else:
            context = "No specific web context available. Use your general financial knowledge."
        
        # Enhanced prompt for better responses
        system_prompt = """You are an expert financial analyst and research assistant.

Your role:
- Provide accurate, insightful financial analysis based on PROVIDED DATA
- ALWAYS use real-time stock price data when provided in the context
- Structure responses clearly with headers and sections
- Include specific numbers, dates, and metrics from the data
- Be concise but comprehensive
- Use professional but accessible language
- NEVER say "data not available" if stock prices are in the context

CRITICAL: If stock price data is provided in the context, YOU MUST include it in your response.

Response format:
- Start with a direct answer using the provided data
- Use markdown formatting (###, **, bullet points)
- Cite key facts with actual numbers
- Provide actionable insights when relevant"""

        user_prompt = f"""Research Query: {query}

Available Information:
{context}

IMPORTANT: Use the stock price data provided above to answer this query completely. 
If multiple companies are mentioned, provide data for ALL of them.
Do not say information is unavailable if it's in the context above.

Provide a well-structured, professional analysis answering this query."""

        answer = await self.llm.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.7,
            max_tokens=1200
        )
        
        return answer
    
    async def _fetch_stock_prices_from_query(self, query: str) -> str:
        """Extract company names and fetch real stock prices using yfinance"""
        import yfinance as yf
        from datetime import datetime
        
        # Common company name to ticker mapping
        ticker_map = {
            'microsoft': 'MSFT', 'msft': 'MSFT',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'googl': 'GOOGL',
            'apple': 'AAPL', 'aapl': 'AAPL',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'meta': 'META', 'facebook': 'META',
            'netflix': 'NFLX', 'nflx': 'NFLX',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'amd': 'AMD',
            'intel': 'INTC', 'intc': 'INTC',
            'oracle': 'ORCL', 'orcl': 'ORCL',
            'ibm': 'IBM',
            'salesforce': 'CRM', 'crm': 'CRM',
            'adobe': 'ADBE', 'adbe': 'ADBE',
            'cisco': 'CSCO', 'csco': 'CSCO',
            'paypal': 'PYPL', 'pypl': 'PYPL',
        }
        
        query_lower = query.lower()
        found_tickers = []
        
        # Extract tickers from query
        for company, ticker in ticker_map.items():
            if company in query_lower:
                if ticker not in found_tickers:
                    found_tickers.append(ticker)
        
        if not found_tickers:
            logger.info("No recognized tickers in query")
            return ""
        
        logger.info(f"Found tickers: {found_tickers}")
        
        # Fetch stock data
        stock_data_parts = []
        today = datetime.now().strftime("%B %d, %Y")
        
        for ticker in found_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Get current/latest price data
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                open_price = info.get('regularMarketOpen') or info.get('open')
                previous_close = info.get('previousClose')
                day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
                day_low = info.get('dayLow') or info.get('regularMarketDayLow')
                volume = info.get('volume') or info.get('regularMarketVolume')
                market_cap = info.get('marketCap')
                company_name = info.get('longName') or info.get('shortName') or ticker
                
                # Build comprehensive stock data
                data = f"""
**{company_name} ({ticker}) - Real-Time Stock Data**
Date: {today}

Current Price: ${current_price:.2f} USD
Opening Price: ${open_price:.2f} USD
Previous Close: ${previous_close:.2f} USD
Day Range: ${day_low:.2f} - ${day_high:.2f} USD
Volume: {volume:,} shares
Market Cap: ${market_cap:,} USD

Price Change: ${current_price - previous_close:.2f} ({((current_price - previous_close) / previous_close * 100):.2f}%)
"""
                stock_data_parts.append(data)
                logger.info(f"Successfully fetched data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                # Add partial data if available
                stock_data_parts.append(f"\n**{ticker}**: Unable to fetch real-time data\n")
        
        if stock_data_parts:
            return "\n".join(stock_data_parts)
        
        return ""
    
    async def _fetch_stock_prices_from_company_info(self, company_info: Dict) -> str:
        """Fetch stock prices using validated company info from Universal Resolver"""
        import yfinance as yf
        from datetime import datetime
        
        stock_data_parts = []
        today = datetime.now().strftime("%B %d, %Y")
        
        # Handle multiple companies
        if company_info.get('multiple'):
            companies = company_info.get('companies', [])
            logger.info(f"Fetching data for {len(companies)} companies")
            
            for comp in companies:
                ticker = comp.get('ticker')
                if not ticker:
                    continue
                
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Validate that we got valid data
                    if not info or not isinstance(info, dict):
                        raise ValueError(f"No data returned for {ticker}")
                    
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    open_price = info.get('regularMarketOpen') or info.get('open')
                    previous_close = info.get('previousClose')
                    day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
                    day_low = info.get('dayLow') or info.get('regularMarketDayLow')
                    volume = info.get('volume') or info.get('regularMarketVolume')
                    market_cap = info.get('marketCap')
                    company_name = comp.get('company', ticker)
                    currency = comp.get('currency', 'USD')
                    
                    # Validate essential data
                    if current_price is None or previous_close is None:
                        raise ValueError(f"Missing essential price data for {ticker}")
                    
                    # Calculate price change safely
                    price_change = current_price - previous_close
                    price_change_pct = (price_change / previous_close * 100) if previous_close > 0 else 0
                    
                    # Format values with proper conditionals
                    market_cap_str = f"{market_cap:,}" if market_cap else 'N/A'
                    open_price_str = f"{currency} {open_price:.2f}" if open_price else 'N/A'
                    day_low_str = f"{currency} {day_low:.2f}" if day_low else 'N/A'
                    day_high_str = f"{currency} {day_high:.2f}" if day_high else 'N/A'
                    volume_str = f"{volume:,}" if volume else 'N/A'
                    
                    data = f"""
**{company_name} ({ticker}) - Real-Time Stock Data**
Date: {today}
Exchange: {comp.get('country', 'N/A')}

Current Price: {currency} {current_price:.2f}
Opening Price: {open_price_str}
Previous Close: {currency} {previous_close:.2f}
Day Range: {day_low_str} - {day_high_str}
Volume: {volume_str} shares
Market Cap: {currency} {market_cap_str}

Price Change: {currency} {price_change:.2f} ({price_change_pct:.2f}%)
Sector: {comp.get('sector', 'N/A')}
"""
                    stock_data_parts.append(data)
                    logger.info(f"Fetched data for {ticker}")
                    
                except ValueError as ve:
                    logger.warning(f"Data validation error for {ticker}: {ve}")
                    stock_data_parts.append(f"\n**{ticker}**: {str(ve)}\n")
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                    stock_data_parts.append(f"\n**{ticker}**: Unable to fetch real-time data\n")
        
        else:
            # Single company
            ticker = company_info.get('ticker')
            if ticker:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    open_price = info.get('regularMarketOpen') or info.get('open')
                    previous_close = info.get('previousClose')
                    day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
                    day_low = info.get('dayLow') or info.get('regularMarketDayLow')
                    volume = info.get('volume') or info.get('regularMarketVolume')
                    market_cap = info.get('marketCap')
                    company_name = company_info.get('company', ticker)
                    currency = company_info.get('currency', 'USD')
                    
                    # Add note if this is a subsidiary resolution
                    subsidiary_note = ""
                    if company_info.get('is_subsidiary'):
                        original = company_info.get('original_mention', '')
                        subsidiary_note = f"\nNote: '{original.title()}' is a subsidiary/division of {company_name}\n"
                    
                    data = f"""
**{company_name} ({ticker}) - Real-Time Stock Data**
Date: {today}
Exchange: {company_info.get('exchange', 'N/A')} ({company_info.get('country', 'N/A')}){subsidiary_note}

Current Price: {currency} {current_price:.2f}
Opening Price: {currency} {open_price:.2f}
Previous Close: {currency} {previous_close:.2f}
Day Range: {currency} {day_low:.2f} - {currency} {day_high:.2f}
Volume: {volume:,} shares
Market Cap: {currency} {market_cap:,}

Price Change: {currency} {current_price - previous_close:.2f} ({((current_price - previous_close) / previous_close * 100):.2f}%)
Sector: {company_info.get('sector', 'N/A')}
Industry: {company_info.get('industry', 'N/A')}

About: {company_info.get('description', 'N/A')[:300]}...
"""
                    stock_data_parts.append(data)
                    logger.info(f"Fetched detailed data for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                    stock_data_parts.append(f"\n**{ticker}**: Unable to fetch real-time data\n")
        
        if stock_data_parts:
            return "\n".join(stock_data_parts)
        
        return ""
    
    def clear_context(self):
        """Clear the vector store"""
        self.vector_store.clear_store()
        logger.info("Vector store cleared")
    
    async def _extract_companies_from_text(self, text: str) -> List[str]:
        """Extract pharmaceutical and other company names/tickers from text using LLM"""
        try:
            prompt = f"""Extract all pharmaceutical and biotech company names and stock tickers from this text.

Text: {text}

Return a JSON array of tickers ONLY. Examples: ["ABBV", "LLY", "JNJ"]
For pharmaceutical companies like:
- AbbVie → ABBV
- Eli Lilly → LLY
- Johnson & Johnson → JNJ
- Pfizer → PFE
- Merck → MRK
- Bristol Myers Squibb → BMY
- Amgen → AMGN
- Gilead → GILD
- Regeneron → REGN
- Moderna → MRNA

Return ONLY the JSON array, nothing else."""

            response = await self.llm.generate_response(prompt)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                tickers = json.loads(json_match.group(0))
                # Validate and filter
                valid_tickers = [t.upper() for t in tickers if isinstance(t, str) and len(t) <= 5]
                logger.info(f"Extracted tickers from text: {valid_tickers}")
                return valid_tickers
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting companies from text: {e}")
            return []
    
    async def _fetch_multiple_stock_prices(self, tickers: List[str]) -> str:
        """Fetch stock prices for multiple tickers"""
        import yfinance as yf
        from datetime import datetime
        
        if not tickers:
            return ""
        
        stock_data_parts = []
        today = datetime.now().strftime("%B %d, %Y")
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Validate data exists
                if not info or not isinstance(info, dict):
                    logger.warning(f"No data returned for {ticker}")
                    continue
                
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                open_price = info.get('regularMarketOpen') or info.get('open')
                previous_close = info.get('previousClose')
                day_high = info.get('dayHigh') or info.get('regularMarketDayHigh')
                day_low = info.get('dayLow') or info.get('regularMarketDayLow')
                volume = info.get('volume') or info.get('regularMarketVolume')
                market_cap = info.get('marketCap')
                company_name = info.get('longName') or info.get('shortName') or ticker
                sector = info.get('sector', 'N/A')
                
                # Check if we have essential data
                if current_price is None:
                    logger.warning(f"No price data for {ticker}")
                    continue
                
                # Calculate price change safely
                if previous_close and previous_close > 0:
                    price_change = current_price - previous_close
                    price_change_pct = (price_change / previous_close * 100)
                else:
                    price_change = 0
                    price_change_pct = 0
                
                # Format values with proper conditionals
                market_cap_str = f"${market_cap:,}" if market_cap else 'N/A'
                open_price_str = f"${open_price:.2f}" if open_price else 'N/A'
                prev_close_str = f"${previous_close:.2f}" if previous_close else 'N/A'
                day_low_str = f"${day_low:.2f}" if day_low else 'N/A'
                day_high_str = f"${day_high:.2f}" if day_high else 'N/A'
                volume_str = f"{volume:,}" if volume else 'N/A'
                
                data = f"""
**{company_name} ({ticker}) - Real-Time Stock Data**
Date: {today}
Sector: {sector}

Current Price: ${current_price:.2f} USD
Opening Price: {open_price_str} USD
Previous Close: {prev_close_str} USD
Day Range: {day_low_str} - {day_high_str} USD
Volume: {volume_str} shares
Market Cap: {market_cap_str}

Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
"""
                stock_data_parts.append(data)
                logger.info(f"Fetched stock data for {ticker}: ${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        if stock_data_parts:
            return "\n".join(stock_data_parts)
        
        return ""
    
    async def _try_calculate(self, query: str, context_chunks: List[str]) -> Optional[str]:
        """
        Detect calculation queries and perform actual math using FinancialCalculator
        
        Returns calculation result string if this is a calculation query, None otherwise
        """
        query_lower = query.lower()
        
        # Calculation detection keywords
        calc_keywords = ['calculate', 'compute', 'what is', 'growth rate', 'yoy', 'year over year',
                         'margin', 'profit margin', 'net margin', 'return on', 'roi', 'cagr',
                         'compound annual', 'percentage', 'how much', 'difference between']
        
        is_calculation = any(keyword in query_lower for keyword in calc_keywords)
        
        if not is_calculation:
            return None
        
        logger.info("Detected calculation query - attempting to extract financial data")
        
        # Combine all context
        full_context = query + "\n" + "\n".join(context_chunks[:5])
        
        # Determine what metric we're calculating
        metric_type = self._identify_metric_type(query_lower)
        logger.info(f"Identified metric type: {metric_type}")
        
        # Extract relevant financial data based on metric type
        financial_data = self._extract_financial_data(full_context, metric_type)
        
        if not financial_data or len(financial_data) < 2:
            logger.info(f"Not enough {metric_type} data found for calculation")
            return None
        
        # Determine calculation type and perform
        if 'yoy' in query_lower or 'year over year' in query_lower or 'growth' in query_lower:
            # Extract specific years from query if mentioned
            import re
            year_matches = re.findall(r'\b(20\d{2})\b', query)
            
            if len(year_matches) >= 2:
                # User specified years - use them
                requested_years = [int(y) for y in year_matches[:2]]
                requested_years.sort(reverse=True)  # Most recent first
                
                # Find data for these specific years
                data_dict = {year: value for year, value in financial_data}
                
                if requested_years[0] in data_dict and requested_years[1] in data_dict:
                    current_period = requested_years[0]
                    current_value = data_dict[current_period]
                    previous_period = requested_years[1]
                    previous_value = data_dict[previous_period]
                    logger.info(f"Using user-specified years: {current_period} vs {previous_period}")
                else:
                    # Requested years not found, use most recent
                    logger.warning(f"Requested years {requested_years} not found in data")
                    sorted_data = sorted(financial_data, key=lambda x: x[0], reverse=True)
                    current_period, current_value = sorted_data[0]
                    previous_period, previous_value = sorted_data[1]
            else:
                # No specific years mentioned, use most recent two
                sorted_data = sorted(financial_data, key=lambda x: x[0], reverse=True)
                current_period, current_value = sorted_data[0]
                previous_period, previous_value = sorted_data[1]
            
            logger.info(f"Calculating YoY growth: {current_period}=${current_value:,.2f}, {previous_period}=${previous_value:,.2f}")
            
            result = self.calculator.calculate_growth_rate(
                current_value=current_value,
                previous_value=previous_value
            )
            
            calc_response = f"""
### 📊 Year-over-Year Growth Calculation

**Metric:** {metric_type.title()}

**Formula:** {result.formula}

**Inputs:**
- Current Period ({current_period}): ${current_value:,.2f} billion
- Previous Period ({previous_period}): ${previous_value:,.2f} billion

**Result:** **{result.result}%** growth

**Interpretation:** {result.interpretation}

*Note: This is a programmatic calculation performed using Python, not an LLM estimate.*
"""
            return calc_response
        
        elif 'margin' in query_lower:
            # Need profit and revenue
            if len(financial_data) >= 2:
                profit, revenue = financial_data[0][1], financial_data[1][1]
                logger.info(f"Calculating margin: profit=${profit:,.2f}, revenue=${revenue:,.2f}")
                
                result = self.calculator.calculate_margin(
                    profit=profit,
                    revenue=revenue,
                    margin_type='net'
                )
                
                calc_response = f"""
### 📈 Profit Margin Calculation

**Formula:** {result.formula}

**Inputs:**
- Profit: ${profit:,.2f}
- Revenue: ${revenue:,.2f}

**Result:** **{result.result}%**

**Interpretation:** The net profit margin is {result.result}%, indicating profitability relative to revenue.

*Note: This is a programmatic calculation performed using Python, not an LLM estimate.*
"""
                return calc_response
        
        elif 'cagr' in query_lower or 'compound annual' in query_lower:
            # Need starting, ending, and years
            if len(financial_data) >= 2:
                sorted_data = sorted(financial_data, key=lambda x: x[0])
                start_year, start_value = sorted_data[0]
                end_year, end_value = sorted_data[-1]
                num_years = end_year - start_year
                
                logger.info(f"Calculating CAGR: {start_year}=${start_value:,.2f}, {end_year}=${end_value:,.2f}, years={num_years}")
                
                result = self.calculator.calculate_cagr(
                    starting_value=start_value,
                    ending_value=end_value,
                    num_years=num_years
                )
                
                calc_response = f"""
### 📈 Compound Annual Growth Rate (CAGR)

**Formula:** {result.formula}

**Inputs:**
- Starting Value ({start_year}): ${start_value:,.2f}
- Ending Value ({end_year}): ${end_value:,.2f}
- Number of Years: {num_years}

**Result:** **{result.result}%**

**Interpretation:** {result.interpretation}

*Note: This is a programmatic calculation performed using Python, not an LLM estimate.*
"""
                return calc_response
        
        # If we didn't match a specific calculation type
        return None
    
    def _identify_metric_type(self, query_lower: str) -> str:
        """Identify what financial metric is being queried"""
        if 'revenue' in query_lower or 'sales' in query_lower:
            return 'revenue'
        elif 'earnings' in query_lower or 'net income' in query_lower or 'profit' in query_lower:
            return 'earnings'
        elif 'ebitda' in query_lower:
            return 'ebitda'
        elif 'stock price' in query_lower or 'share price' in query_lower:
            return 'stock_price'
        else:
            return 'revenue'  # Default
    
    def _is_calculation_query(self, query: str) -> bool:
        """Check if query is asking for a calculation"""
        query_lower = query.lower()
        calc_keywords = ['calculate', 'compute', 'what is', 'growth rate', 'yoy', 'year over year',
                         'margin', 'profit margin', 'net margin', 'return on', 'roi', 'cagr',
                         'compound annual', 'percentage', 'how much', 'difference between']
        return any(keyword in query_lower for keyword in calc_keywords)
    
    async def _search_for_financial_data(self, query: str, company_info: Optional[Dict]) -> str:
        """
        Perform targeted search for financial data needed for calculations
        Returns formatted financial data string
        """
        import re
        
        # Extract company name
        company_name = None
        if company_info and company_info.get('resolved'):
            company_name = company_info.get('company', '')
        
        if not company_name:
            # Try to extract from query
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ['inc', 'inc.', 'corp', 'corp.', 'ltd', 'ltd.']:
                    if i > 0:
                        company_name = ' '.join(words[max(0, i-2):i+1])
                        break
        
        if not company_name:
            logger.warning("Could not identify company for financial search")
            return ""
        
        # Determine what metric we need
        metric_type = self._identify_metric_type(query.lower())
        
        # Extract years from query - use user-specified years if available
        requested_years = re.findall(r'\b(20\d{2})\b', query)
        
        # Build targeted search queries based on user's requested years
        if len(requested_years) >= 2:
            # User specified years - search for those specific years
            years = sorted([int(y) for y in requested_years[:2]], reverse=True)
            search_queries = [
                f"{company_name} {metric_type} {years[0]} {years[1]}",
                f"{company_name} annual {metric_type} {years[1]}-{years[0]}",
                f"{company_name} {metric_type} historical data",
            ]
            logger.info(f"🔍 Searching for {company_name} {metric_type} data for years: {years}")
        elif len(requested_years) == 1:
            # Single year - get that year and previous
            year = int(requested_years[0])
            search_queries = [
                f"{company_name} {metric_type} {year} {year-1}",
                f"{company_name} annual {metric_type} {year}",
                f"{company_name} {metric_type} historical",
            ]
            logger.info(f"🔍 Searching for {company_name} {metric_type} data for {year}")
        else:
            # No specific years - search for recent data
            current_year = 2026
            search_queries = [
                f"{company_name} {metric_type} 2024 2023 2022",
                f"{company_name} annual {metric_type} historical",
                f"{company_name} {metric_type} by year",
            ]
            logger.info(f"🔍 Searching for {company_name} recent {metric_type} data")
        
        logger.info(f"Searching for {company_name} {metric_type} data")
        
        # Try searches until we find data
        for search_query in search_queries:
            try:
                search_results = await self.search_client.search(search_query, max_results=5)
                
                if search_results:
                    logger.info(f"Found {len(search_results)} results for financial data")
                    
                    # Scrape top 2 results
                    scraped_docs = await self._scrape_results_parallel(search_results[:2])
                    
                    if scraped_docs:
                        # Combine scraped content
                        combined_text = "\n\n".join([doc.content for doc in scraped_docs])
                        
                        # Try to extract financial data
                        financial_data = self._extract_financial_data(combined_text, metric_type)
                        
                        if len(financial_data) >= 2:
                            # Format as context
                            data_lines = [f"{year}: ${value:.2f} billion" for year, value in financial_data[:3]]
                            context = f"\n**{company_name} {metric_type.title()} Data:**\n" + "\n".join(data_lines)
                            logger.info(f"Successfully extracted {len(financial_data)} data points")
                            return context
            except Exception as e:
                logger.error(f"Error in financial search: {e}")
                continue
        
        logger.warning(f"Could not find sufficient {metric_type} data for {company_name}")
        return ""
    
    def _extract_financial_data(self, text: str, metric_type: str) -> List[tuple]:
        """
        Extract financial data with year/period labels
        Returns list of (year, value) tuples
        """
        import re
        
        financial_data = []
        
        # Patterns to match financial data with years (tested and working)
        patterns = [
            # Pattern: "FY2024 revenue was $391.04 billion" or "2024 revenue was $95.3 billion"
            r'(?:FY\s*)?(\d{4})\s+(?:revenue|earnings|sales|profit|income|ebitda)\s+(?:was|is|of)\s+\$\s*([\d,\.]+)\s+(?:billion|B)',
            # Pattern: "revenue: $95.3 billion (2024)" or similar with year at end
            r'(?:revenue|earnings|sales|profit|income|ebitda)[:\s]+\$\s*([\d,\.]+)\s+(?:billion|B).*?(\d{4})',
            # Pattern: "2024: $95.3B" or "FY 2024: $95.3 billion"
            r'(?:FY\s*)?(\d{4})[:\s]+\$?\s*([\d,\.]+)\s*(?:billion|B)',
            # Pattern: "in 2024, revenue of $95.3 billion"
            r'in\s+(?:FY\s*)?(\d{4}),?\s+(?:revenue|earnings|sales)\s+(?:of|was|is)\s+\$\s*([\d,\.]+)\s+(?:billion|B)',
            # Pattern: "$95.3 billion in revenue for 2024"
            r'\$\s*([\d,\.]+)\s+(?:billion|B)\s+(?:in|of)?\s*(?:revenue|earnings|sales)?\s+(?:for|in)\s+(?:FY\s*)?(\d{4})',
            # Pattern: "FY 2024 Total Revenue $130.49B" (Nvidia format)
            r'(?:FY\s*)?(\d{4})\s+(?:Total\s+)?(?:Revenue|Earnings|Sales)\s+\$\s*([\d,\.]+)\s*(?:billion|B)',
            # Pattern: "Revenue (TTM) $130.49B 2024" (reverse order with TTM)
            r'(?:Revenue|Earnings|Sales)\s+(?:\(TTM\))?\s+\$\s*([\d,\.]+)\s*(?:billion|B).*?(\d{4})',
            # Pattern: "2023 - $26.97B" or "2024 - Revenue: $130B"
            r'(\d{4})\s*[-:]\s*(?:Revenue|Earnings|Sales)?[:\s]*\$\s*([\d,\.]+)\s*(?:billion|B)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    if len(groups) == 2:
                        g1, g2 = groups
                        # Determine which group is year and which is value
                        # Check if g1 is a 4-digit year
                        if g1.replace(',', '').replace('.', '').isdigit() and len(g1) == 4:
                            year = int(g1)
                            value = float(g2.replace(',', ''))
                        # Check if g2 is a 4-digit year
                        elif g2.replace(',', '').replace('.', '').isdigit() and len(g2) == 4:
                            year = int(g2)
                            value = float(g1.replace(',', ''))
                        else:
                            continue
                        
                        # Validate year and value ranges
                        if 2000 <= year <= 2030 and value > 0.1:
                            financial_data.append((year, value))
                            logger.info(f"Found {metric_type} data: {year} = ${value:.2f}B")
                except Exception as e:
                    logger.debug(f"Error parsing match: {e}")
                    continue
        
        # Remove duplicates and sort by year (most recent first)
        financial_data = list(set(financial_data))
        financial_data.sort(key=lambda x: x[0], reverse=True)
        
        return financial_data
    
    # =========================================================================
    # DEEP RESEARCH MODE - Phase 1-3 Implementation
    # =========================================================================
    
    async def deep_research_mode(self, query: str, company_info: Optional[Dict] = None, max_iterations: int = 15) -> ResearchResult:
        """Execute deep iterative research with multiple exploration steps
        
        This implements true \"Deep Research\" mode with:
        - 5-15 iterative research loops
        - Progressive deepening based on discoveries
        - Intelligent follow-up question generation
        - Comprehensive synthesis of all findings
        """
        logger.info(f"🔬 Starting Deep Research Mode for: {query}")
        logger.info(f"   Max iterations: {max_iterations}")
        
        # Initialize research state
        research_state = {
            "query": query,
            "findings": [],
            "follow_up_questions": [],
            "explored_topics": set(),
            "all_sources": [],
            "iteration": 0,
            "company_info": company_info
        }
        
        # Add initial query as first follow-up
        research_state["follow_up_questions"].append(query)
        
        # Iterative research loop
        while research_state["iteration"] < max_iterations:
            iteration_num = research_state["iteration"] + 1
            logger.info(f"\\n{'='*80}")
            logger.info(f"🔬 Deep Research Iteration {iteration_num}/{max_iterations}")
            logger.info(f"{'='*80}")
            
            # Check if we have questions to explore
            if not research_state["follow_up_questions"]:
                logger.info("No more follow-up questions, generating new ones...")
                if research_state["iteration"] > 0:
                    follow_ups = await self._generate_follow_up_questions(research_state)
                    if not follow_ups:
                        logger.info("✅ No new research directions found, concluding research")
                        break
                    research_state["follow_up_questions"].extend(follow_ups)
                else:
                    break
            
            # Get next research question
            current_query = research_state["follow_up_questions"].pop(0)
            research_state["explored_topics"].add(current_query.lower()[:50])
            
            logger.info(f"📍 Current research focus: {current_query}")
            
            # Execute research step
            try:
                # Search and scrape (increased to 6 results for more sources)
                search_results = await self.search_client.search(current_query, max_results=6)
                
                if search_results:
                    # Store sources (deduplicate by URL)
                    existing_urls = {s.get('url', '') for s in research_state["all_sources"]}
                    for sr in search_results[:6]:  # Collect up to 6 sources per iteration
                        if sr.get('url') and sr['url'] not in existing_urls:
                            research_state["all_sources"].append(sr)
                            existing_urls.add(sr['url'])
                    
                    # Scrape top results
                    scraped_docs = await self._scrape_results_parallel(search_results[:3])
                    
                    if scraped_docs:
                        # Extract insights from scraped content
                        insights = await self._extract_insights(scraped_docs, current_query)
                        
                        # Store findings
                        research_state["findings"].append({
                            "iteration": research_state["iteration"],
                            "query": current_query,
                            "insights": insights,
                            "doc_count": len(scraped_docs)
                        })
                        
                        logger.info(f"✅ Extracted insights: {insights[:200]}...")
                    else:
                        logger.warning(f"⚠️ No documents scraped for: {current_query}")
                else:
                    logger.warning(f"⚠️ No search results for: {current_query}")
                
            except Exception as e:
                logger.error(f"❌ Error in iteration {iteration_num}: {e}")
            
            research_state["iteration"] += 1
            
            # Check if research is sufficient (after minimum 5 iterations)
            if research_state["iteration"] >= 5:
                is_sufficient = await self._is_research_sufficient(research_state)
                if is_sufficient:
                    logger.info(f"✅ Research deemed sufficient after {research_state['iteration']} iterations")
                    break
        
        logger.info(f"\\n{'='*80}")
        logger.info(f"🎯 Deep Research Complete: {research_state['iteration']} iterations")
        logger.info(f"📊 Total findings: {len(research_state['findings'])}")
        logger.info(f"📚 Total sources: {len(research_state['all_sources'])}")
        logger.info(f"{'='*80}\\n")
        
        # Synthesize all findings into comprehensive report
        final_result = await self._synthesize_research(research_state)
        
        return final_result
    
    async def _generate_follow_up_questions(self, state: dict) -> List[str]:
        """Generate intelligent follow-up questions based on current findings
        
        This is the \"Research Intelligence\" - adapting based on discoveries
        """
        if not state["findings"]:
            return []
        
        # Get recent findings (last 3)
        recent_findings = state["findings"][-3:]
        findings_summary = "\\n\\n".join([
            f"Research {i+1}: {f['query']}\\nInsights: {f['insights'][:300]}"
            for i, f in enumerate(recent_findings)
        ])
        
        prompt = f"""You are a financial research analyst conducting deep research on: {state['query']}

Recent research findings:
{findings_summary}

Based on these findings, generate 3 specific follow-up questions to deepen the research.

Focus on:
- Unexplored aspects or companies mentioned in findings
- Deeper dives into interesting trends or numbers
- Quantitative data needs (revenue, growth, market share)
- Competitive analysis or comparisons
- Recent news or developments mentioned

Avoid questions about topics already explored: {', '.join(list(state['explored_topics'])[:5])}

Return ONLY a JSON array of 3 questions, nothing else:
["question 1", "question 2", "question 3"]"""

        try:
            response = await self.llm.generate_response(prompt, temperature=0.8, max_tokens=300)
            
            # Extract JSON array
            import json
            # Try to find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                questions = json.loads(json_str)
                
                # Filter out already explored topics
                new_questions = []
                for q in questions:
                    q_lower = q.lower()[:50]
                    if q_lower not in state["explored_topics"]:
                        new_questions.append(q)
                
                logger.info(f"Generated {len(new_questions)} new follow-up questions")
                return new_questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
        
        return []
    
    async def _extract_insights(self, documents: List[Document], query: str) -> str:
        """Extract key insights and identify new research directions"""
        if not documents:
            return "No documents to analyze"
        
        # Combine document contents (limit to avoid token overflow)
        combined_text = "\\n\\n".join([
            doc.content[:1500] for doc in documents[:3]
        ])
        
        prompt = f"""Analyze this content from research on: {query}

Content:
{combined_text}

Extract and summarize:
1. Key facts, data points, and numbers (revenue, growth, market size, etc.)
2. Important trends or patterns mentioned
3. Companies, products, or entities that need deeper investigation
4. Interesting developments or recent news

Be concise but capture all important details. Focus on financial and business insights.

Provide 3-5 concise bullet points."""

        try:
            insights = await self.llm.generate_response(prompt, temperature=0.3, max_tokens=400)
            return insights
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return "Unable to extract insights"
    
    async def _is_research_sufficient(self, state: dict) -> bool:
        """Determine if research has sufficient depth and breadth"""
        # Minimum iteration requirement
        if state["iteration"] < 5:
            return False
        
        # If we've done 12+ iterations, that's usually enough
        if state["iteration"] >= 12:
            logger.info("Research complete: Maximum iteration threshold reached")
            return True
        
        # Check if we have substantial findings
        if len(state["findings"]) < 4:
            return False
        
        # Ask LLM to evaluate completeness
        findings_summary = "\\n".join([
            f"- {f['query']}: {f['insights'][:150]}"
            for f in state["findings"][-5:]
        ])
        
        prompt = f"""Evaluate research completeness for: {state['query']}

Research iterations: {state['iteration']}
Findings gathered: {len(state['findings'])}
Sources collected: {len(state['all_sources'])}

Recent findings:
{findings_summary}

Question: Is this research comprehensive enough for a deep analysis report covering multiple perspectives, data points, and insights?

Answer with just: yes or no"""

        try:
            response = await self.llm.generate_response(prompt, temperature=0.2, max_tokens=10)
            is_sufficient = "yes" in response.lower()
            
            if is_sufficient:
                logger.info("LLM evaluation: Research is comprehensive")
            else:
                logger.info("LLM evaluation: More research needed")
            
            return is_sufficient
        except Exception as e:
            logger.error(f"Error evaluating research sufficiency: {e}")
            # Default to continue if we can't evaluate
            return False
    
    async def _synthesize_research(self, state: dict) -> ResearchResult:
        """Synthesize all research findings into comprehensive report"""
        logger.info(f"📝 Synthesizing {len(state['findings'])} research findings...")
        
        # Fetch stock data if company info available
        stock_context = ""
        if state.get("company_info") and state["company_info"].get("resolved"):
            stock_context = await self._fetch_stock_prices_from_company_info(state["company_info"])
            logger.info(f"Added stock price context: {len(stock_context)} chars")
        
        # Combine all insights
        all_insights = "\\n\\n".join([
            f"**Research Step {f['iteration'] + 1}**: {f['query']}\\n{f['insights']}"
            for f in state["findings"]
        ])
        
        synthesis_prompt = f"""You are a senior financial analyst creating a comprehensive deep research report.

ORIGINAL QUERY: {state['query']}

You have conducted {len(state['findings'])} iterations of research. Synthesize ALL findings below into a thorough, professional report.

{'='*80}
RESEARCH FINDINGS ({len(state['findings'])} iterations):
{'='*80}

{all_insights}

{'='*80}
REAL-TIME STOCK DATA:
{'='*80}
{stock_context if stock_context else 'No stock data available'}

{'='*80}

Create a comprehensive financial research report with:

1. **Executive Summary** (2-3 paragraphs overview)
2. **Key Findings** (synthesize main discoveries from ALL research iterations)
3. **Detailed Analysis** (organize by themes discovered during research)
4. **Financial Metrics & Data** (include ALL numbers, percentages, trends found)
5. **Market Dynamics** (competitive landscape, trends, challenges)
6. **Forward-Looking Insights** (projections, opportunities, risks)
7. **Conclusion** (actionable summary)

IMPORTANT:
- Use ALL information gathered across {len(state['findings'])} research steps
- Include specific numbers, percentages, and data points
- Cite companies, products, and entities mentioned
- Be thorough - aim for 1500-2000 words
- Use professional markdown formatting
- If stock data is provided, prominently feature it

Write a detailed, well-structured report:"""

        try:
            final_answer = await self.llm.generate_with_system_prompt(
                system_prompt="You are a senior financial analyst creating comprehensive deep research reports with meticulous attention to detail.",
                user_message=synthesis_prompt,
                temperature=0.4,
                max_tokens=3500
            )
            
            logger.info(f"✅ Synthesis complete: {len(final_answer)} chars")
            
            # Build research reasoning trail
            research_trail = [
                {
                    "action": f"Researched: {f['query']}",
                    "reasoning": f"Iteration {f['iteration'] + 1}: {f['insights'][:200]}...",
                    "doc_count": f.get('doc_count', 0)
                }
                for f in state["findings"]
            ]
            
            # Deduplicate sources by URL and ensure we have at least 5 unique sources
            seen_urls = set()
            unique_sources = []
            for source in state["all_sources"]:
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            
            # Log source collection stats
            logger.info(f"📚 Collected {len(unique_sources)} unique sources from {len(state['all_sources'])} total")
            
            # Ensure minimum 5 sources for deep research
            if len(unique_sources) < 5:
                logger.warning(f"⚠️ Only {len(unique_sources)} unique sources found, expected 5+")
            
            return ResearchResult(
                answer=final_answer,
                sources=unique_sources[:15],  # Top 15 unique sources
                query=state["query"],
                context_used=[],
                research_reasoning=research_trail,
                iteration_count=state["iteration"]
            )
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            # Fallback to basic summary
            return ResearchResult(
                answer=f"Deep research completed with {len(state['findings'])} iterations. " + all_insights[:1000],
                sources=state["all_sources"][:10],
                query=state["query"],
                context_used=[],
                research_reasoning=[],
                iteration_count=state["iteration"]
            )
    
    def _has_inline_values(self, query: str) -> bool:
        """
        Check if query contains numeric values (indicating calculation with provided data)
        """
        import re
        
        # Calculation keywords
        calc_keywords = ['calculate', 'compute', 'find', 'what is', 'growth rate', 
                         'margin', 'ratio', 'roe', 'roi', 'p/e', 'pe ratio', 'cagr']
        
        query_lower = query.lower()
        has_calc_keyword = any(keyword in query_lower for keyword in calc_keywords)
        
        if not has_calc_keyword:
            return False
        
        # Check for numeric values
        # Pattern for numbers like: 2000, 1500, 5000 cr, 25000 crores, 80, etc.
        number_patterns = [
            r'\b\d{1,5}\b',  # Simple numbers (80, 2000, etc.)
            r'\d+\s*(?:cr|crore|crores|billion|million|thousand)\b',  # Numbers with units
            r'\b\d+\.\d+\b',  # Decimals
        ]
        
        # Count how many numbers found
        numbers_found = 0
        for pattern in number_patterns:
            matches = re.findall(pattern, query_lower)
            numbers_found += len(matches)
        
        # If we have calculation keyword AND at least 2 numbers, it's inline calculation
        return numbers_found >= 2
    
    async def _calculate_with_inline_values(self, query: str) -> Optional[str]:
        """
        🚀 HYBRID CALCULATION ENGINE
        
        Type 1: Direct calculations (values provided) - Pure Python math
        Type 2: Fetch + Calculate (company mentioned) - yfinance  
        Type 3: LLM fallback for uncommon metrics
        
        Returns formatted calculation result or None
        """
        import re
        
        query_lower = query.lower()
        
        # ============================================================
        # STEP 1: Check for Type 2 (Company + Metric, No Values)
        # ============================================================
        company_name = self._extract_company_name_from_query(query)
        
        if company_name:
            # Check if query has inline numbers (Type 1) or needs fetching (Type 2)
            numbers = self._extract_numbers_from_query(query)
            
            if len(numbers) < 2:  # Type 2: No values provided, fetch from company
                logger.info(f"🏢 Type 2: Company '{company_name}' detected without inline values")
                
                # Determine which metric is requested
                metric = None
                if 'roe' in query_lower or 'return on equity' in query_lower:
                    metric = 'roe'
                elif 'pe' in query_lower or 'p/e' in query_lower or 'price to earnings' in query_lower:
                    metric = 'pe_ratio'
                elif 'debt' in query_lower and 'equity' in query_lower:
                    metric = 'debt_to_equity'
                elif 'profit margin' in query_lower or 'net margin' in query_lower:
                    metric = 'profit_margin'
                elif 'current ratio' in query_lower:
                    metric = 'current_ratio'
                
                if metric:
                    try:
                        result = self.calculator.calculate_metric_for_company(company_name, metric)
                        if result:
                            return self._format_company_metric_result(result, company_name)
                        else:
                            logger.warning(f"Could not fetch {metric} for {company_name}")
                            return f"❌ Could not fetch {metric} data for {company_name}. The company may not be in our database or data is unavailable.\n\nTry providing values manually or check if yfinance is installed."
                    except Exception as e:
                        logger.error(f"Error in Type 2 calculation: {e}")
                        return f"❌ Error fetching data for {company_name}: {str(e)}\n\nPlease try again or provide values manually."
                
                # If company found but no metric matched, return error instead of falling through
                return f"❌ Could not determine which metric to calculate for {company_name}. Please specify the metric (P/E, ROE, etc.)."
            
            # If company found AND has 2+ numbers, proceed with Type 1 using those values
            else:
                logger.info(f"📊 Company '{company_name}' detected WITH inline values - using Type 1 calculation")
        
        # ============================================================
        # STEP 2: Extract Numbers for Type 1 (Direct Calculation)
        # ============================================================
        numbers = self._extract_numbers_from_query(query)
        logger.info(f"Extracted {len(numbers)} numbers from query: {numbers}")
        
        if len(numbers) < 2:
            logger.info("Not enough numbers for Type 1 calculation")
            return None
        
        # ============================================================
        # STEP 3: Type 1 - Direct Calculations (Top 10 Hardcoded)
        # ============================================================
        
        try:
            # CAGR (needs 3 numbers - check FIRST before Growth Rate)
            if ('cagr' in query_lower or 'compound annual' in query_lower) and len(numbers) >= 3:
                logger.info("📊 Type 1: Calculating CAGR")
                
                # Smart extraction for CAGR parameters
                import re
                initial = None
                final = None
                years = None
                
                # Clean query for better matching
                clean_query = query.replace('$', '').replace('₹', '').replace('€', '').replace('£', '')
                clean_lower = clean_query.lower()
                
                # Try to find initial value
                initial_match = re.search(r'(?:initial|starting|start|begin|from)\s*(?:value|amount|is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if initial_match:
                    initial = float(initial_match.group(1).replace(',', ''))
                
                # Try to find final value
                final_match = re.search(r'(?:final|ending|end|to)\s*(?:value|amount|is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if final_match:
                    final = float(final_match.group(1).replace(',', ''))
                
                # Try to find time period
                years_match = re.search(r'(?:period|years?|time)\s*(?:is|of|=|:)?\s*(\d+)', clean_lower)
                if years_match:
                    years = int(years_match.group(1))
                
                # Fallback: use order from query (first number = initial, second = final, third = years)
                if initial is None and len(numbers) >= 1:
                    initial = numbers[0]
                if final is None and len(numbers) >= 2:
                    final = numbers[1]
                if years is None and len(numbers) >= 3:
                    years = int(numbers[2])
                
                logger.info(f"CAGR calculation: initial={initial}, final={final}, years={years}")
                result = self.calculator.calculate_cagr(initial, final, years)
                return await self._format_calculation_result(result, 'CAGR', query)
            
            # P/E Ratio
            elif 'p/e' in query_lower or 'pe ratio' in query_lower or 'price to earnings' in query_lower or 'price-to-earnings' in query_lower:
                logger.info("📊 Type 1: Calculating P/E Ratio")
                
                # Smart extraction: look for keywords to identify which number is which
                import re
                price = None
                eps = None
                
                # Clean query for better matching (remove currency symbols)
                clean_query = query.replace('$', '').replace('₹', '').replace('€', '').replace('£', '')
                clean_lower = clean_query.lower()
                
                # Try to find price near "price" or "stock" keywords
                price_match = re.search(r'(?:price|stock)\s*(?:is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if price_match:
                    price = float(price_match.group(1).replace(',', ''))
                
                # Try to find EPS near "eps" or "earnings" keywords  
                eps_match = re.search(r'(?:eps|earnings per share|earnings)\s*(?:is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if eps_match:
                    eps = float(eps_match.group(1).replace(',', ''))
                
                # If we couldn't find them by keywords, use size heuristic
                if price is None or eps is None:
                    if len(numbers) >= 2:
                        # Larger number is usually price
                        price = max(numbers[0], numbers[1])
                        eps = min(numbers[0], numbers[1])
                    else:
                        return None
                
                logger.info(f"P/E calculation: price={price}, eps={eps}")
                result = self.calculator.calculate_pe_ratio(price, eps)
                return await self._format_calculation_result(result, 'P/E Ratio', query)
            
            # ROE
            elif 'roe' in query_lower or 'return on equity' in query_lower:
                logger.info("📊 Type 1: Calculating ROE")
                
                # Smart extraction: look for keywords
                import re
                net_income = None
                equity = None
                
                # Clean query for better matching
                clean_query = query.replace('$', '').replace('₹', '').replace('€', '').replace('£', '')
                clean_lower = clean_query.lower()
                
                # Try to find net income
                income_match = re.search(r'(?:net income|income|profit|earnings)\s*(?:is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if income_match:
                    net_income = float(income_match.group(1).replace(',', ''))
                
                # Try to find equity
                equity_match = re.search(r'(?:equity|shareholder equity|shareholders equity)\s*(?:is|of|=|:)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)', clean_lower)
                if equity_match:
                    equity = float(equity_match.group(1).replace(',', ''))
                
                # Fallback: smaller number is usually net income
                if net_income is None or equity is None:
                    if len(numbers) >= 2:
                        net_income = min(numbers[0], numbers[1])
                        equity = max(numbers[0], numbers[1])
                    else:
                        return None
                
                logger.info(f"ROE calculation: net_income={net_income}, equity={equity}")
                result = self.calculator.calculate_roe(net_income, equity)
                return await self._format_calculation_result(result, 'ROE', query)
            
            # Debt-to-Equity (NEW)
            elif 'debt' in query_lower and 'equity' in query_lower:
                logger.info("📊 Type 1: Calculating Debt-to-Equity")
                debt, equity = numbers[0], numbers[1]
                result = self.calculator.calculate_debt_to_equity(debt, equity)
                return await self._format_calculation_result(result, 'Debt-to-Equity', query)
            
            # Profit Margin (NEW)
            elif 'profit margin' in query_lower or ('margin' in query_lower and 'profit' in query_lower):
                logger.info("📊 Type 1: Calculating Profit Margin")
                profit, revenue = numbers[0], numbers[1]
                result = self.calculator.calculate_margin(profit, revenue, 'net')
                return await self._format_calculation_result(result, 'Profit Margin', query)
            
            # Current Ratio (NEW)
            elif 'current ratio' in query_lower:
                logger.info("📊 Type 1: Calculating Current Ratio")
                current_assets, current_liabilities = numbers[0], numbers[1]
                result = self.calculator.calculate_current_ratio(current_assets, current_liabilities)
                return await self._format_calculation_result(result, 'Current Ratio', query)
            
            # Quick Ratio (NEW - needs 3 numbers)
            elif 'quick ratio' in query_lower and len(numbers) >= 3:
                logger.info("📊 Type 1: Calculating Quick Ratio")
                current_assets, inventory, current_liabilities = numbers[0], numbers[1], numbers[2]
                result = self.calculator.calculate_quick_ratio(current_assets, inventory, current_liabilities)
                return await self._format_calculation_result(result, 'Quick Ratio', query)
            
            # Operating Margin (NEW)
            elif 'operating margin' in query_lower:
                logger.info("📊 Type 1: Calculating Operating Margin")
                operating_income, revenue = numbers[0], numbers[1]
                result = self.calculator.calculate_margin(operating_income, revenue, 'operating')
                return await self._format_calculation_result(result, 'Operating Margin', query)
            
            # Growth Rate (check AFTER CAGR)
            elif 'growth' in query_lower or 'growth rate' in query_lower or 'yoy' in query_lower:
                logger.info("📊 Type 1: Calculating Growth Rate")
                old_value, new_value = numbers[0], numbers[1]
                result = self.calculator.calculate_growth_rate(new_value, old_value)
                return await self._format_calculation_result(result, 'Growth Rate', query)
            
            # ============================================================
            # STEP 4: LLM Fallback for Uncommon Calculations (NEW)
            # ============================================================
            else:
                logger.info("⚡ Type 3: No hardcoded match - using LLM fallback")
                return await self._llm_fallback_calculation(query, numbers)
        
        except ValueError as e:
            logger.error(f"Calculation error: {e}")
            return f"❌ Calculation error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in calculation: {e}")
            return None
    
    def _extract_company_name_from_query(self, query: str) -> Optional[str]:
        """Extract company name from query for Type 2 calculations"""
        query_lower = query.lower()
        
        # Check against supported companies from calculator
        if hasattr(self.calculator, 'TICKER_MAP'):
            for company in self.calculator.TICKER_MAP.keys():
                if company in query_lower:
                    logger.info(f"✅ Extracted company: {company}")
                    return company
        
        return None
    
    def _extract_numbers_from_query(self, query: str) -> List[float]:
        """Extract all numbers from query (optimized with better year detection)"""
        import re
        numbers = []
        query_lower = query.lower()
        
        # Remove currency symbols to avoid interfering with number extraction
        clean_query = query.replace('$', '').replace('₹', '').replace('€', '').replace('£', '')
        
        # Pattern 1: Numbers with "crores" or "cr"  
        crore_pattern = r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:cr|crore|crores)\b'
        crore_matches = re.findall(crore_pattern, query_lower)
        for match in crore_matches:
            num = float(match.replace(',', ''))
            if num not in numbers:
                numbers.append(num)
        
        # Pattern 2: All numeric values (with optional commas and decimals)
        # Match numbers with thousand separators (1,000 or 10,000) or regular numbers
        number_pattern = r'\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\b'
        all_matches = re.findall(number_pattern, clean_query)
        
        for match in all_matches:
            # Remove commas and convert to float
            num = float(match.replace(',', ''))
            
            # Skip if already added
            if num in numbers:
                continue
            
            # Smart year detection: only exclude if it's clearly a year
            if 1900 <= num <= 2030 and num == int(num):
                # Find the position of this number in the original query
                # Check context around it for year indicators
                match_str = str(int(num))
               
                if match_str in clean_query or match in clean_query:
                    pos = clean_query.find(match if match in clean_query else match_str)
                    # Get context around the number (30 chars before and after)
                    start = max(0, pos - 30)
                    end = min(len(clean_query), pos + len(match) + 30)
                    context = clean_query[start:end].lower()
                    
                    # Year indicators to check
                    year_keywords = [
                        'year', 'fy', 'cy', 'in 20', 'for 20', 'during 20',
                        'from 20', 'to 20', 'by 20', 'since 20', 'until 20',
                        'q1 20', 'q2 20', 'q3 20', 'q4 20'
                    ]
                    
                    # If we find year-related keywords, skip this number
                    if any(keyword in context for keyword in year_keywords):
                        logger.debug(f"Skipping {num} - detected as year from context: {context}")
                        continue
            
            numbers.append(num)
        
        logger.info(f"Extracted numbers from query: {numbers}")
        return numbers
    
    async def _format_calculation_result(
        self, 
        result,  # CalculationResult from FinancialCalculator
        metric_name: str,
        query: str = ""  # Original query to detect currency
    ) -> str:
        """Format Type 1 (direct calculation) result using OpenAI for professional output"""
        
        # Detect currency from query
        currency_symbol = ""
        if '$' in query:
            currency_symbol = "$"
        elif '₹' in query or 'rupee' in query.lower() or 'inr' in query.lower():
            currency_symbol = "₹"
        elif '€' in query or 'euro' in query.lower():
            currency_symbol = "€"
        elif '£' in query or 'pound' in query.lower() or 'gbp' in query.lower():
            currency_symbol = "£"
        # If no currency detected, leave empty - don't assume
        
        # Build prompt for OpenAI to format professionally (matching Type 2 style)
        prompt = f"""You are a professional financial analyst. Format this {metric_name} calculation in a clean, professional manner.

**Calculation Data:**
- Metric: {metric_name}
- Formula: {result.formula}
- Input Values: {result.inputs}
- Result: {result.result} {result.unit}
- Interpretation: {result.interpretation}
- Currency: {"Use " + currency_symbol + " for currency values" if currency_symbol else "Do NOT use any currency symbol - show plain numbers"}

**Required Format:**
Use clear headings and bullet points. NO box characters (┏━━┃╔══║). Keep it clean and professional.

**Structure:**
1. Start with metric name as heading
2. Show key input data as bullet points  
3. Calculation steps with formula
4. Final result (bold/highlighted)
5. Professional interpretation with insights
6. Source note: "Source: Financial Calculator (Direct Computation)"

**Style Guidelines:**
- Use bullet points (•) for lists
- Use emojis sparingly (📊 for metrics, ✅ for results)
- {currency_symbol + " for values" if currency_symbol else "Plain numbers without currency symbols"}
- Keep concise but informative
- Professional tone
- Show calculation steps clearly
- DO NOT include dates in source
- Source MUST be: "Source: Financial Calculator (Direct Computation)"
- Maximum 250 words"""

        try:
            formatted = await self.llm.generate_response(prompt)
            return formatted
        except Exception as e:
            logger.error(f"LLM formatting error: {e}")
            # Fallback to simple bullet format
            output = f"\n## 📊 {metric_name}\n\n"
            output += "**Input Data:**\n"
            if result.inputs:
                for key, value in result.inputs.items():
                    key_display = key.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        output += f"  • {key_display}: {currency_symbol}{value:,.2f}\n" if currency_symbol else f"  • {key_display}: {value:,.2f}\n"
                    else:
                        output += f"  • {key_display}: {value}\n"
            output += f"\n**Formula:** {result.formula}\n\n"
            output += f"**Result:** {result.result} {result.unit}\n\n"
            output += f"**Analysis:** {result.interpretation}\n\n"
            output += "*Source: Financial Calculator (Direct Computation)*\n"
            return output
    
    def _format_company_metric_result(self, result, company_name: str) -> str:
        """Format Type 2 (company fetch) result"""
        
        metric_name = result.calculation_type.replace('_company', '').replace('_', ' ').title()
        
        output = f"""
### 📊 {metric_name} Analysis for {company_name.upper()}

**Company:** {company_name.upper()} ({result.inputs.get('ticker', 'N/A')})

**{metric_name}:** {result.result} {result.unit if result.unit != 'ratio' else ''}

**Formula:** {result.formula}

**Interpretation:** {result.interpretation}

**Source:** Yahoo Finance (Real-time Market Data)
"""
        return output
    
    async def _llm_fallback_calculation(self, query: str, numbers: List[float]) -> str:
        """Type 3: LLM-powered dynamic calculation for uncommon metrics"""
        
        logger.info(f"Using LLM fallback for uncommon calculation with {len(numbers)} numbers")
        
        prompt = f"""You are a financial calculator.

Query: "{query}"
Numbers: {numbers}

Task:
1. Identify the financial metric
2. Determine the formula
3. Map numbers to variables
4. Calculate the result
5. Provide interpretation

Format:
- Metric: [name]
- Formula: [formula]
- Calculation: [steps]
- Result: [answer with units]
- Interpretation: [explanation]

Max 150 words. Be precise."""

        try:
            response = await self.llm.generate_response(prompt)
            return f"### 📊 Dynamic Calculation\n\n{response}\n\n*AI-assisted calculation*"
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return None
    
    async def _handle_fetch_and_calculate(
        self, 
        query: str,
        company_info: Dict
    ) -> Optional[ResearchResult]:
        """
        TYPE 2 CALCULATION: Fetch financial data and then calculate
        Example: "Calculate P/E for TCS", "What is Reliance ROE?"
        """
        
        logger.info("📊 Starting fetch and calculate workflow")
        
        # Step 1: Identify calculation type
        calc_type = self._identify_calculation_type_from_query(query)
        if not calc_type:
            logger.warning("Could not identify calculation type")
            return None
        
        logger.info(f"Identified calculation type: {calc_type}")
        
        # Step 2: Get required metrics for this calculation
        required_metrics = self._get_required_metrics_for_calculation(calc_type)
        if not required_metrics:
            logger.warning(f"No metric mapping for {calc_type}")
            return None
        
        logger.info(f"Required metrics: {required_metrics}")
        
        # Step 3: Handle single or multiple companies
        calculation_results = []
        
        if company_info.get('multiple'):
            # Multiple companies - comparison
            companies = company_info.get('companies', [])
            logger.info(f"Processing {len(companies)} companies for comparison")
            
            for company in companies:
                result = await self._fetch_and_calculate_single(
                    company, 
                    calc_type, 
                    required_metrics
                )
                if result:
                    calculation_results.append(result)
        
        else:
            # Single company
            company = {
                'company': company_info.get('company'),
                'ticker': company_info.get('ticker'),
                'sector': company_info.get('sector'),
                'currency': company_info.get('currency', 'USD')
            }
            result = await self._fetch_and_calculate_single(
                company, 
                calc_type, 
                required_metrics
            )
            if result:
                calculation_results.append(result)
        
        # Step 4: Format response
        if not calculation_results:
            logger.warning("No calculation results obtained")
            return None
        
        # Format based on single or comparison
        if len(calculation_results) == 1:
            answer = await self._format_single_calculation(calculation_results[0])
        else:
            answer = await self._format_comparative_calculation(calculation_results, calc_type)
        
        return ResearchResult(
            query=query,
            answer=answer,
            sources=[{
                'url': 'https://finance.yahoo.com',
                'title': 'Yahoo Finance - Real-time Financial Data',
                'snippet': f'Financial metrics for {calc_type} calculation'
            }],
            context_used=["Yahoo Finance API"],
            iteration_count=1,
            research_reasoning=[{
                'action': f'Fetched financial data and calculated {calc_type}',
                'reasoning': 'Used Yahoo Finance API to retrieve real-time financial metrics'
            }]
        )
    
    async def _fetch_and_calculate_single(
        self,
        company: Dict,
        calc_type: str,
        required_metrics: List[str]
    ) -> Optional[Dict]:
        """
        Fetch data and calculate for a single company
        """
        
        ticker = company.get('ticker')
        company_name = company.get('company', ticker)
        
        logger.info(f"Fetching data for {company_name} ({ticker})")
        
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract required metrics
            financial_data = {}
            for metric in required_metrics:
                value = info.get(metric)
                if value is not None:
                    financial_data[metric] = value
                else:
                    logger.warning(f"Metric '{metric}' not available for {ticker}")
            
            # Check if we have all required data
            if len(financial_data) < len(required_metrics):
                logger.warning(f"Missing some metrics for {ticker}")
                # Try alternative methods...
                if calc_type == 'ROE' and 'returnOnEquity' in info:
                    # Yahoo Finance already has calculated ROE
                    financial_data['pre_calculated_roe'] = info['returnOnEquity'] * 100
            
            if not financial_data:
                logger.error(f"No financial data available for {ticker}")
                return None
            
            # Perform calculation
            result = self._calculate_from_fetched_data(
                calc_type,
                financial_data,
                company_name,
                company.get('currency', 'USD')
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching/calculating for {ticker}: {e}")
            return None
    
    def _identify_calculation_type_from_query(self, query: str) -> Optional[str]:
        """
        Identify what type of calculation is being requested
        """
        query_lower = query.lower()
        
        # Map keywords to calculation types
        if 'p/e' in query_lower or 'pe ratio' in query_lower or 'price to earnings' in query_lower or 'price-to-earnings' in query_lower:
            return 'P/E Ratio'
        elif 'roe' in query_lower or 'return on equity' in query_lower:
            return 'ROE'
        elif 'roa' in query_lower or 'return on assets' in query_lower:
            return 'ROA'
        elif 'debt to equity' in query_lower or 'debt-to-equity' in query_lower or 'd/e ratio' in query_lower:
            return 'Debt-to-Equity'
        elif 'current ratio' in query_lower:
            return 'Current Ratio'
        elif 'quick ratio' in query_lower or 'acid test' in query_lower:
            return 'Quick Ratio'
        elif 'profit margin' in query_lower or 'net margin' in query_lower:
            return 'Profit Margin'
        elif 'dividend yield' in query_lower:
            return 'Dividend Yield'
        elif 'eps' in query_lower or 'earnings per share' in query_lower:
            return 'EPS'
        elif 'book value' in query_lower:
            return 'Book Value'
        elif 'market cap' in query_lower or 'market capitalization' in query_lower:
            return 'Market Cap'
        
        return None
    
    def _get_required_metrics_for_calculation(self, calc_type: str) -> List[str]:
        """
        Map calculation types to required Yahoo Finance metric keys
        """
        
        metric_map = {
            'P/E Ratio': ['currentPrice', 'trailingEps'],
            'ROE': ['returnOnEquity'],  # Yahoo has this pre-calculated
            'ROA': ['returnOnAssets'],   # Yahoo has this pre-calculated
            'Debt-to-Equity': ['debtToEquity'],  # Yahoo has this pre-calculated
            'Current Ratio': ['currentRatio'],   # Yahoo has this pre-calculated
            'Quick Ratio': ['quickRatio'],       # Yahoo has this pre-calculated
            'Profit Margin': ['profitMargins'],  # Yahoo has this pre-calculated
            'Dividend Yield': ['dividendYield'], # Yahoo has this pre-calculated
            'EPS': ['trailingEps'],
            'Book Value': ['bookValue'],
            'Market Cap': ['marketCap']
        }
        
        return metric_map.get(calc_type, [])
    
    def _calculate_from_fetched_data(
        self,
        calc_type: str,
        financial_data: Dict,
        company_name: str,
        currency: str = 'USD'
    ) -> Dict:
        """
        Perform calculation using fetched financial data
        """
        
        result = {
            'company': company_name,
            'calculation_type': calc_type,
            'inputs': {},
            'result': None,
            'interpretation': '',
            'currency': currency
        }
        
        try:
            if calc_type == 'P/E Ratio':
                price = financial_data.get('currentPrice')
                eps = financial_data.get('trailingEps')
                
                if price and eps and eps != 0:
                    pe_ratio = round(price / eps, 2)
                    result['inputs'] = {'price': price, 'eps': eps}
                    result['result'] = pe_ratio
                    result['interpretation'] = self._interpret_pe_ratio(pe_ratio)
                else:
                    result['error'] = 'Missing price or EPS data'
            
            elif calc_type == 'ROE':
                # Check if Yahoo has pre-calculated ROE
                roe = financial_data.get('pre_calculated_roe') or financial_data.get('returnOnEquity')
                
                if roe:
                    # Convert to percentage if needed
                    if roe < 1:
                        roe = roe * 100
                    roe = round(roe, 2)
                    result['inputs'] = {'roe': roe}
                    result['result'] = f"{roe}%"
                    result['interpretation'] = self._interpret_roe(roe)
                else:
                    result['error'] = 'ROE data not available'
            
            elif calc_type == 'ROA':
                roa = financial_data.get('returnOnAssets')
                if roa:
                    if roa < 1:
                        roa = roa * 100
                    roa = round(roa, 2)
                    result['inputs'] = {'roa': roa}
                    result['result'] = f"{roa}%"
                    result['interpretation'] = self._interpret_roa(roa)
                else:
                    result['error'] = 'ROA data not available'
            
            elif calc_type == 'Debt-to-Equity':
                de_ratio = financial_data.get('debtToEquity')
                if de_ratio:
                    de_ratio = round(de_ratio, 2)
                    result['inputs'] = {'debt_to_equity': de_ratio}
                    result['result'] = de_ratio
                    result['interpretation'] = self._interpret_debt_to_equity(de_ratio)
                else:
                    result['error'] = 'Debt-to-Equity data not available'
            
            elif calc_type == 'Current Ratio':
                current_ratio = financial_data.get('currentRatio')
                if current_ratio:
                    current_ratio = round(current_ratio, 2)
                    result['inputs'] = {'current_ratio': current_ratio}
                    result['result'] = current_ratio
                    result['interpretation'] = self._interpret_current_ratio(current_ratio)
                else:
                    result['error'] = 'Current Ratio data not available'
            
            elif calc_type == 'Quick Ratio':
                quick_ratio = financial_data.get('quickRatio')
                if quick_ratio:
                    quick_ratio = round(quick_ratio, 2)
                    result['inputs'] = {'quick_ratio': quick_ratio}
                    result['result'] = quick_ratio
                    result['interpretation'] = self._interpret_quick_ratio(quick_ratio)
                else:
                    result['error'] = 'Quick Ratio data not available'
            
            elif calc_type == 'Profit Margin':
                margin = financial_data.get('profitMargins')
                if margin:
                    if margin < 1:
                        margin = margin * 100
                    margin = round(margin, 2)
                    result['inputs'] = {'profit_margin': margin}
                    result['result'] = f"{margin}%"
                    result['interpretation'] = self._interpret_profit_margin(margin)
                else:
                    result['error'] = 'Profit Margin data not available'
            
            elif calc_type == 'Dividend Yield':
                div_yield = financial_data.get('dividendYield')
                if div_yield:
                    if div_yield < 1:
                        div_yield = div_yield * 100
                    div_yield = round(div_yield, 2)
                    result['inputs'] = {'dividend_yield': div_yield}
                    result['result'] = f"{div_yield}%"
                    result['interpretation'] = self._interpret_dividend_yield(div_yield)
                else:
                    result['error'] = 'Dividend Yield data not available'
            
            elif calc_type == 'EPS':
                eps = financial_data.get('trailingEps')
                if eps:
                    eps = round(eps, 2)
                    result['inputs'] = {'eps': eps}
                    result['result'] = f"{currency} {eps:.2f}"
                    result['interpretation'] = f"Trailing 12-month earnings per share"
                else:
                    result['error'] = 'EPS data not available'
            
            elif calc_type == 'Book Value':
                book_value = financial_data.get('bookValue')
                if book_value:
                    book_value = round(book_value, 2)
                    result['inputs'] = {'book_value': book_value}
                    result['result'] = f"{currency} {book_value:.2f}"
                    result['interpretation'] = f"Book value per share"
                else:
                    result['error'] = 'Book Value data not available'
            
            elif calc_type == 'Market Cap':
                market_cap = financial_data.get('marketCap')
                if market_cap:
                    result['inputs'] = {'market_cap': market_cap}
                    result['result'] = f"{currency} {market_cap:,.0f}"
                    result['interpretation'] = f"Total market capitalization"
                else:
                    result['error'] = 'Market Cap data not available'
        
        except Exception as e:
            logger.error(f"Calculation error for {calc_type}: {e}")
            result['error'] = str(e)
        
        return result
    
    async def _format_single_calculation(self, calc_result: Dict) -> str:
        """
        Format calculation result for a single company using OpenAI for professional output
        """
        
        if calc_result.get('error'):
            return f"❌ Could not calculate {calc_result['calculation_type']} for {calc_result['company']}: {calc_result['error']}"
        
        company = calc_result['company']
        calc_type = calc_result['calculation_type']
        inputs = calc_result['inputs']
        result = calc_result['result']
        interpretation = calc_result['interpretation']
        currency = calc_result.get('currency', 'USD')
        ticker = inputs.get('ticker', 'N/A')
        
        # Build prompt for OpenAI to format professionally
        prompt = f"""You are a professional financial analyst. Format this {calc_type} analysis for {company} in a clean, professional manner.

**Data:**
- Company: {company}
- Ticker: {ticker}
- Metric: {calc_type}
- Result: {result}
- Currency: {currency}
- Input Values: {inputs}
- Interpretation: {interpretation}

**Required Format:**
Use clear headings and bullet points. NO box characters (┏━━┃). Keep it clean and professional.

**Structure:**
1. Start with company name and metric as heading
2. Show key input data as bullet points
3. Formula and calculation steps (if applicable)
4. Final result (bold/highlighted)
5. Professional interpretation with insights
6. Source note at the end

**Style Guidelines:**
- Use bullet points (•) for lists
- Use emojis sparingly (📊 for metrics, ✅ for results)
- Keep concise but informative
- Professional tone
- Maximum 250 words"""

        try:
            formatted = await self.llm.generate_response(prompt)
            return formatted
        except Exception as e:
            logger.error(f"LLM formatting error: {e}")
            # Fallback to simple bullet format
            output = f"\n## 📊 {calc_type} - {company}\n\n"
            output += f"**Company:** {company} ({ticker})\n\n"
            output += "**Input Data:**\n"
            for key, value in inputs.items():
                if isinstance(value, (int, float)):
                    output += f"  • {key.replace('_', ' ').title()}: {currency} {value:,.2f}\n"
                else:
                    output += f"  • {key.replace('_', ' ').title()}: {value}\n"
            output += f"\n**Result:** {result}\n\n"
            output += f"**Analysis:** {interpretation}\n\n"
            output += "*Source: Yahoo Finance (Real-time data)*\n"
            return output
    
    async def _format_comparative_calculation(self, calc_results: List[Dict], calc_type: str) -> str:
        """
        Format calculation results for multiple companies (comparison) using OpenAI for professional output
        """
        
        # Prepare data for OpenAI formatting
        companies_data = []
        for result in calc_results:
            companies_data.append({
                'company': result['company'],
                'ticker': result.get('inputs', {}).get('ticker', 'N/A'),
                'result': result.get('result', 'N/A'),
                'interpretation': result.get('interpretation', ''),
                'error': result.get('error', None)
            })
        
        # Build prompt for OpenAI
        prompt = f"""You are a professional financial analyst. Create a comparative analysis of {calc_type} for multiple companies.

**Companies Data:**
{companies_data}

**Required Format:**
Use clear headings and bullet points. NO box characters (┏━━┃╔══║). Keep it clean and professional.

**Structure:**
1. Title: "Comparative {calc_type} Analysis"
2. Summary comparison using bullet points
3. Individual company details with:
   - Company name and ticker
   - {calc_type} value
   - Brief interpretation
4. Key insights and recommendations
5. Source note

**Style Guidelines:**
- Use bullet points (•) for lists
- Use emojis sparingly (📊 for metrics, ✅/⚠️ for status)
- Professional tone
- Highlight winners/losers in comparison
- Maximum 350 words"""

        try:
            formatted = await self.llm.generate_response(prompt)
            return formatted
        except Exception as e:
            logger.error(f"LLM formatting error for comparison: {e}")
            # Fallback to simple bullet format
            response = f"\n## 📊 Comparative {calc_type} Analysis\n\n"
            response += "**Summary:**\n"
            for result in calc_results:
                status = '✅' if not result.get('error') else '❌'
                response += f"  • {result['company']}: {result.get('result', 'N/A')} {status}\n"
            
            response += "\n**Detailed Analysis:**\n\n"
            for i, result in enumerate(calc_results, 1):
                response += f"{i}. **{result['company']}**\n"
                if result.get('error'):
                    response += f"   - ❌ Error: {result['error']}\n\n"
                else:
                    response += f"   - {calc_type}: {result['result']}\n"
                    response += f"   - Analysis: {result['interpretation']}\n\n"
            
            response += "*Source: Yahoo Finance (Real-time data)*\n"
            return response
    
    # Interpretation helper methods
    def _interpret_pe_ratio(self, pe: float) -> str:
        if pe < 0:
            return "⚠️ Negative P/E indicates the company is currently unprofitable."
        elif pe < 15:
            return f"💰 P/E of {pe} is relatively low, suggesting the stock may be undervalued or the company has low growth expectations."
        elif pe <= 25:
            return f"✅ P/E of {pe} is moderate, indicating fair valuation for a mature company."
        else:
            return f"🚀 P/E of {pe} is high, suggesting investors expect strong growth or the stock may be overvalued."
    
    def _interpret_roe(self, roe: float) -> str:
        if roe < 0:
            return "⚠️ Negative ROE indicates the company is unprofitable."
        elif roe < 10:
            return f"📉 ROE of {roe}% is below average, indicating weak profitability."
        elif roe <= 20:
            return f"✅ ROE of {roe}% is good, indicating healthy profitability."
        else:
            return f"🌟 ROE of {roe}% is excellent, indicating very strong profitability and efficient use of equity."
    
    def _interpret_roa(self, roa: float) -> str:
        if roa < 0:
            return "⚠️ Negative ROA indicates the company is unprofitable."
        elif roa < 5:
            return f"📉 ROA of {roa}% is low, indicating inefficient use of assets."
        elif roa <= 10:
            return f"✅ ROA of {roa}% is good, indicating efficient asset utilization."
        else:
            return f"🌟 ROA of {roa}% is excellent, indicating highly efficient asset management."
    
    def _interpret_debt_to_equity(self, ratio: float) -> str:
        if ratio < 0.5:
            return f"💪 D/E ratio of {ratio} is low, indicating conservative use of debt and strong financial stability."
        elif ratio <= 1.5:
            return f"✅ D/E ratio of {ratio} is moderate, indicating balanced leverage."
        else:
            return f"⚠️ D/E ratio of {ratio} is high, indicating heavy reliance on debt which may increase financial risk."
    
    def _interpret_current_ratio(self, ratio: float) -> str:
        if ratio < 1:
            return f"⚠️ Current ratio of {ratio} is below 1, indicating potential liquidity concerns."
        elif ratio <= 2:
            return f"✅ Current ratio of {ratio} is healthy, indicating good short-term financial health."
        else:
            return f"💰 Current ratio of {ratio} is very high, indicating strong liquidity but possibly inefficient use of assets."
    
    def _interpret_quick_ratio(self, ratio: float) -> str:
        if ratio < 0.5:
            return f"⚠️ Quick ratio of {ratio} is low, indicating potential liquidity issues."
        elif ratio <= 1:
            return f"✅ Quick ratio of {ratio} is adequate, indicating reasonable liquidity."
        else:
            return f"💪 Quick ratio of {ratio} is strong, indicating excellent short-term liquidity."
    
    def _interpret_profit_margin(self, margin: float) -> str:
        if margin < 0:
            return "⚠️ Negative profit margin indicates the company is operating at a loss."
        elif margin < 5:
            return f"📉 Profit margin of {margin}% is low, indicating thin profitability."
        elif margin <= 15:
            return f"✅ Profit margin of {margin}% is good, indicating healthy profitability."
        else:
            return f"🌟 Profit margin of {margin}% is excellent, indicating very strong profitability."
    
    def _interpret_dividend_yield(self, div_yield: float) -> str:
        if div_yield == 0:
            return "ℹ️ Company does not currently pay dividends."
        elif div_yield < 2:
            return f"📊 Dividend yield of {div_yield}% is low, typical of growth-focused companies."
        elif div_yield <= 5:
            return f"✅ Dividend yield of {div_yield}% is moderate, providing steady income."
        else:
            return f"💰 Dividend yield of {div_yield}% is high, indicating strong dividend payments (verify sustainability)."
    
    async def _extract_company_from_query(self, query: str) -> Optional[Dict]:
        """
        Extract company information from query for calculation purposes
        Returns company_info dict with resolved=True if successful
        """
        import yfinance as yf
        
        # Known Indian company tickers (most common ones)
        indian_companies = {
            'tcs': 'TCS.NS',
            'tata consultancy': 'TCS.NS',
            'infosys': 'INFY.NS',
            'wipro': 'WIPRO.NS',
            'reliance': 'RELIANCE.NS',
            'reliance industries': 'RELIANCE.NS',
            'hdfc bank': 'HDFCBANK.NS',
            'hdfc': 'HDFCBANK.NS',
            'icici bank': 'ICICIBANK.NS',
            'icici': 'ICICIBANK.NS',
            'sbi': 'SBIN.NS',
            'state bank': 'SBIN.NS',
            'bharti airtel': 'BHARTIARTL.NS',
            'airtel': 'BHARTIARTL.NS',
            'itc': 'ITC.NS',
            'bajaj finance': 'BAJFINANCE.NS',
            'maruti': 'MARUTI.NS',
            'hul': 'HINDUNILVR.NS',
            'hindustan unilever': 'HINDUNILVR.NS',
            'axis bank': 'AXISBANK.NS',
            'kotak': 'KOTAKBANK.NS',
            'kotak mahindra': 'KOTAKBANK.NS',
            'titan': 'TITAN.NS',
            'asian paints': 'ASIANPAINT.NS',
            'sun pharma': 'SUNPHARMA.NS',
            'dr reddy': 'DRREDDY.NS',
            'cipla': 'CIPLA.NS',
            'divis labs': 'DIVISLAB.NS',
            'tech mahindra': 'TECHM.NS',
            'hcl tech': 'HCLTECH.NS',
            'hcl': 'HCLTECH.NS',
            'm&m': 'M&M.NS',
            'mahindra': 'M&M.NS',
            'adani': 'ADANIENT.NS',
            'ltim': 'LTIM.NS',
            'lti': 'LTIM.NS',
            'persistent': 'PERSISTENT.NS',
        }
        
        query_lower = query.lower()
        
        # Check for multiple companies (comparison queries)
        if 'compare' in query_lower or ' and ' in query_lower or ' vs ' in query_lower:
            found_companies = []
            for company_name, ticker in indian_companies.items():
                if company_name in query_lower:
                    # Validate ticker exists
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if info.get('symbol'):
                            found_companies.append({
                                'company': info.get('longName', company_name.title()),
                                'ticker': ticker,
                                'sector': info.get('sector', 'Unknown'),
                                'currency': 'INR' if ticker.endswith('.NS') else 'USD'
                            })
                    except:
                        pass
            
            if len(found_companies) >= 2:
                logger.info(f"Extracted {len(found_companies)} companies for comparison")
                return {
                    'resolved': True,
                    'multiple': True,
                    'companies': found_companies
                }
        
        # Check for single company
        for company_name, ticker in indian_companies.items():
            if company_name in query_lower:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info.get('symbol'):
                        logger.info(f"Extracted company: {company_name.title()} ({ticker})")
                        return {
                            'resolved': True,
                            'multiple': False,
                            'company': info.get('longName', company_name.title()),
                            'ticker': ticker,
                            'sector': info.get('sector', 'Unknown'),
                            'currency': 'INR' if ticker.endswith('.NS') else 'USD'
                        }
                except Exception as e:
                    logger.warning(f"Could not resolve {company_name}: {e}")
                    continue
        
        logger.warning("Could not extract company from query")
        return None
    
    def _is_batch_query(self, query: str) -> bool:
        """
        Detect if query contains multiple calculation requests
        """
        # Check for numbered lists
        if re.search(r'\d+\.\s+', query):
            return True
        
        # Check for bullet points
        if re.search(r'[•\-\*]\s+', query):
            return True
        
        # Check for multiple "calculate" or "find" keywords
        calc_keywords = ['calculate', 'find', 'what is', 'determine', 'compute']
        count = sum(1 for kw in calc_keywords if query.lower().count(kw) > 0)
        if sum(query.lower().count(kw) for kw in calc_keywords) >= 3:
            return True
        
        return False
    
    async def _handle_batch_calculations(self, query: str) -> Optional[ResearchResult]:
        """
        Handle multiple calculation queries in a single input
        """
        logger.info("🔢 Processing batch calculations")
        
        # Split query into individual calculations
        queries = self._split_batch_query(query)
        
        if len(queries) <= 1:
            return None
        
        logger.info(f"Found {len(queries)} calculations in batch")
        
        results = []
        for i, calc_query in enumerate(queries, 1):
            logger.info(f"Processing calculation {i}/{len(queries)}")
            
            try:
                # Process individual calculation
                result = await self.research(calc_query, use_web=True)
                results.append({
                    'index': i,
                    'query': calc_query,
                    'answer': result.answer,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error in calculation {i}: {e}")
                results.append({
                    'index': i,
                    'query': calc_query,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Format batch results
        answer = await self._format_batch_results(results)
        
        return ResearchResult(
            query=query,
            answer=answer,
            sources=[],
            context_used=[f"Batch processing: {len(results)} calculations"],
            iteration_count=1
        )
    
    def _split_batch_query(self, query: str) -> List[str]:
        """Split batch query into individual calculations"""
        
        # Helper to filter valid queries
        def is_valid_query(q: str) -> bool:
            """Check if query is a valid calculation request"""
            q = q.lower()
            # Skip header/intro lines
            if any(skip in q for skip in ['help with', 'need help', 'multiple calculations']):
                return False
            # Must be substantial and contain calculation intent
            return len(q) > 10 and any(word in q for word in ['calculate', 'find', 'what', 'ratio', 'rate'])
        
        # Method 1: Numbered lists
        if re.search(r'\d+\.\s+', query):
            parts = re.split(r'\d+\.\s+', query)
            queries = [p.strip() for p in parts if is_valid_query(p)]
            if len(queries) > 1:
                return queries
        
        # Method 2: Newlines
        if '\n' in query:
            parts = query.split('\n')
            queries = [p.strip() for p in parts if is_valid_query(p)]
            if len(queries) > 1:
                return queries
        
        # Method 3: Multiple calc keywords
        calc_keywords = ['calculate', 'find', 'what is', 'determine']
        positions = []
        for keyword in calc_keywords:
            for match in re.finditer(r'\b' + keyword + r'\b', query, re.IGNORECASE):
                positions.append(match.start())
        
        if len(positions) > 1:
            positions.sort()
            queries = []
            for i in range(len(positions)):
                start = positions[i]
                end = positions[i + 1] if i < len(positions) - 1 else len(query)
                sub_query = query[start:end].strip()
                if len(sub_query) > 10:
                    queries.append(sub_query)
            
            if len(queries) > 1:
                return queries
        
        return [query]
    
    async def _format_batch_results(self, results: List[Dict]) -> str:
        """Format multiple calculation results using OpenAI for professional output"""
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        # Build prompt for OpenAI
        prompt = f"""You are a professional financial analyst. Format this batch calculation report in a clean, professional manner.

**Batch Results Summary:**
- Total Calculations: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}
- Success Rate: {(len(successful)/len(results)*100):.1f}%

**Successful Calculations:**
{[{
    'index': r['index'],
    'query': r['query'],
    'answer': r['answer'][:200]  # Truncate for prompt
} for r in successful]}

**Failed Calculations:**
{[{
    'index': r['index'],
    'query': r['query'],
    'error': r['error']
} for r in failed]}

**Required Format:**
Use clear headings and bullet points. NO box characters (┏━━┃╔══║). Keep it clean and professional.

**Structure:**
1. Title: "Batch Calculation Results"
2. Executive summary with stats
3. List each calculation with:
   - Query
   - Result/Answer (brief)
4. Failed calculations section (if any)
5. Final summary

**Style Guidelines:**
- Use bullet points (•) for lists
- Use emojis sparingly (📊 ✅ ❌)
- Professional tone
- Keep concise
- Maximum 400 words"""

        try:
            formatted = await self.llm.generate_response(prompt)
            return formatted
        except Exception as e:
            logger.error(f"LLM formatting error for batch: {e}")
            # Fallback to simple format
            output = f"\n## 📊 Batch Calculation Results\n\n"
            output += f"**Summary:**\n"
            output += f"  • Total: {len(results)}\n"
            output += f"  • Successful: {len(successful)} ✅\n"
            output += f"  • Failed: {len(failed)} ❌\n"
            output += f"  • Success Rate: {(len(successful)/len(results)*100):.1f}%\n\n"
            
            if successful:
                output += "**Results:**\n\n"
                for result in successful:
                    output += f"{result['index']}. {result['query']}\n"
                    output += f"   {result['answer'][:150]}...\n\n"
            
            if failed:
                output += "**Failed:**\n\n"
                for result in failed:
                    output += f"{result['index']}. {result['query']}\n"
                    output += f"   ❌ Error: {result['error']}\n\n"
            
            return output
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'vector_store': self.vector_store.get_stats(),
            'capabilities': {
                'web_search': True,
                'real_time_data': True,
                'document_processing': True,
                'ai_analysis': True,
                'financial_calculations': True  # NEW
            }
        }
