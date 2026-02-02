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

logger = get_logger("research_agent")

@dataclass
class ResearchResult:
    """Research result with sources"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: List[str]

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
    
    async def research(self, query: str, use_web: bool = True) -> ResearchResult:
        """
        Conduct research on a query with smart handling
        
        Args:
            query: Research question
            use_web: Whether to search web for new information
            
        Returns:
            ResearchResult with answer and sources
        """
        logger.info(f"Starting research for query: {query}")
        
        # Classify query
        query_type = self._classify_query(query)
        
        # Handle quick responses (no web search needed)
        if query_type in ['greeting', 'thanks', 'help']:
            logger.info(f"Quick response for {query_type}")
            return ResearchResult(
                answer=self._get_quick_response(query_type),
                sources=[],
                query=query,
                context_used=[]
            )
        
        # Check if it's a financial query
        if not self._is_financial_query(query):
            # For non-financial queries, provide a gentle redirect
            redirect_response = f"""I'm specifically designed for financial research and market analysis. 

Your question: "{query}"

This doesn't appear to be related to stocks, companies, or financial markets. 

I can help you with:
• Stock prices and performance
• Company financial analysis
• Market trends and insights
• Investment research

Would you like to ask a financial question instead?"""
            
            return ResearchResult(
                answer=redirect_response,
                sources=[],
                query=query,
                context_used=[]
            )
        
        # Proceed with full research
        sources = []
        context_chunks = []
        
        # Step 1: Search web if requested (PARALLEL PROCESSING)
        if use_web:
            logger.info("Searching web for information...")
            search_results = await self.search_client.search(query, max_results=3)
            
            if search_results:
                # Scrape top results in parallel
                scraped_docs = await self._scrape_results_parallel(search_results[:2])
                
                if scraped_docs:
                    # Process and add to vector store
                    chunks = self.doc_processor.process_documents(scraped_docs)
                    self.vector_store.add_chunks(chunks)
                    
                    sources.extend([{
                        'title': sr['title'],
                        'url': sr['url'],
                        'snippet': sr['snippet']
                    } for sr in search_results[:3]])
        
        # Step 2: Retrieve relevant context from vector store
        logger.info("Retrieving relevant context from vector store...")
        retrieved_chunks = self.vector_store.search(query, top_k=5)
        
        if retrieved_chunks:
            context_chunks = [chunk['text'] for chunk in retrieved_chunks]
            logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
        
        # Step 3: Generate answer using LLM
        logger.info("Generating answer...")
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
- Provide accurate, insightful financial analysis
- Use data from provided sources when available
- Structure responses clearly with headers and sections
- Include specific numbers, dates, and metrics
- Be concise but comprehensive
- Use professional but accessible language

Response format:
- Start with a direct answer
- Use markdown formatting (###, **, bullet points)
- Cite key facts
- Provide actionable insights when relevant"""

        user_prompt = f"""Research Query: {query}

Available Information:
{context}

Provide a well-structured, professional analysis answering this query. Use the context information when available, and supplement with your knowledge when needed."""

        answer = await self.llm.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.7,
            max_tokens=1200
        )
        
        return answer
    
    def clear_context(self):
        """Clear the vector store"""
        self.vector_store.clear_store()
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'vector_store': self.vector_store.get_stats(),
            'capabilities': {
                'web_search': True,
                'real_time_data': True,
                'document_processing': True,
                'ai_analysis': True
            }
        }
