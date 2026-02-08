"""
Intelligent Query Router - Routes queries to appropriate sector agents
Rejects non-financial queries and routes financial queries to specialized agents
"""
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries"""
    FINANCIAL = "financial"
    NON_FINANCIAL = "non_financial"
    UNCLEAR = "unclear"


class Sector(str, Enum):
    """Available sectors"""
    IT = "IT"
    PHARMA = "Pharma"
    BANKING = "Banking"
    ENERGY = "Energy"
    MANUFACTURING = "Manufacturing"
    CONSUMER = "Consumer"
    REAL_ESTATE = "Real Estate"
    TELECOM = "Telecom"
    AUTOMOTIVE = "Automotive"
    GENERAL = "General"


class RoutingDecision(BaseModel):
    """Result of routing decision"""
    query_type: QueryType
    sector: Optional[Sector] = None
    confidence: float  # 0-1
    reasoning: str
    should_process: bool
    rejection_message: Optional[str] = None
    suggested_agent: Optional[str] = None
    complexity: Optional[str] = "medium"


class QueryRouter:
    """
    Intelligent router that:
    1. Classifies queries as financial/non-financial
    2. Routes financial queries to appropriate sector agents
    3. Rejects non-financial queries politely
    4. Scalable to 10-15 sectors
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        
        # Define sector keywords for quick classification
        self.sector_keywords = {
            Sector.IT: [
                "software", "cloud", "saas", "technology", "it services",
                "consulting", "digital", "ai", "machine learning", "tech",
                "microsoft", "google", "apple", "amazon", "meta", "oracle",
                "salesforce", "adobe", "tcs", "infosys", "wipro", "cognizant"
            ],
            Sector.PHARMA: [
                "pharmaceutical", "pharma", "drug", "medicine", "biotech",
                "healthcare", "clinical", "fda", "biosimilar", "vaccine",
                "pfizer", "moderna", "johnson", "merck", "novartis", "roche",
                "sun pharma", "dr reddy", "cipla", "lupin"
            ],
            Sector.BANKING: [
                "bank", "banking", "financial services", "fintech", "loan",
                "credit", "mortgage", "insurance", "payment", "npa", "capital adequacy",
                "jpmorgan", "goldman sachs", "wells fargo", "hdfc", "icici",
                "axis bank", "sbi", "kotak"
            ],
            Sector.ENERGY: [
                "oil", "gas", "energy", "power", "electricity", "solar",
                "renewable", "coal", "petroleum", "refinery",
                "exxon", "shell", "bp", "chevron", "reliance", "ongc", "ioc"
            ],
            Sector.AUTOMOTIVE: [
                "auto", "automobile", "car", "vehicle", "ev", "electric vehicle",
                "tesla", "ford", "gm", "toyota", "volkswagen", "tata motors",
                "mahindra", "maruti"
            ],
            Sector.CONSUMER: [
                "fmcg", "retail", "consumer goods", "ecommerce", "shopping",
                "walmart", "amazon retail", "target", "unilever", "nestle",
                "hindustan unilever", "itc", "britannia"
            ]
        }
        
        # Non-financial indicators
        self.non_financial_keywords = [
            "recipe", "cooking", "food preparation", "travel tips",
            "movie recommendation", "book review", "weather",
            "sports score", "celebrity gossip", "health advice",
            "workout routine", "fashion", "dating", "relationship"
        ]
    
    def detect_query_complexity(self, query: str) -> str:
        """
        Detect if query is simple or complex
        
        Returns: "simple", "medium", or "complex"
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple query indicators (direct answers)
        simple_indicators = [
            'stock price', 'current price', 'price of', 'what is',
            'how much', 'share price', 'trading at'
        ]
        
        # Complex query indicators (need deep research)
        complex_indicators = [
            'comprehensive', 'deep dive', 'detailed analysis',
            'compare', 'versus', 'vs', 'analyze',
            'trend', 'outlook', 'forecast', 'valuation',
            'competitive position', 'market position'
        ]
        
        # Check for simple queries
        if any(indicator in query_lower for indicator in simple_indicators):
            if word_count <= 6:  # "tesla stock price" = 3 words
                return "simple"
        
        # Check for complex queries
        if any(indicator in query_lower for indicator in complex_indicators):
            return "complex"
        
        # Check word count
        if word_count <= 5:
            return "simple"
        elif word_count <= 10:
            return "medium"
        else:
            return "complex"
    
    async def route_query(self, query: str) -> RoutingDecision:
        """Route query with complexity detection"""
        self.logger.info(f"Routing query: {query}")
        
        # Detect complexity
        complexity = self.detect_query_complexity(query)
        
        # Quick check for non-financial
        quick_decision = self._quick_classify(query)
        if quick_decision:
            quick_decision.complexity = complexity
            return quick_decision
        
        # LLM classification
        llm_decision = await self._llm_classify(query)
        llm_decision.complexity = complexity
        
        self.logger.info(f"Routing: {llm_decision.query_type} -> {llm_decision.sector} (Complexity: {complexity})")
        return llm_decision
    
    def _quick_classify(self, query: str) -> Optional[RoutingDecision]:
        """
        Fast keyword-based classification
        Returns None if unclear, RoutingDecision if confident
        """
        query_lower = query.lower()
        
        # Check for obvious non-financial queries
        for keyword in self.non_financial_keywords:
            if keyword in query_lower:
                return RoutingDecision(
                    query_type=QueryType.NON_FINANCIAL,
                    confidence=0.95,
                    reasoning=f"Detected non-financial keyword: '{keyword}'",
                    should_process=False,
                    rejection_message=self._generate_rejection_message(query)
                )
        
        # Check for sector matches
        sector_scores = {}
        for sector, keywords in self.sector_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                sector_scores[sector] = score
        
        # If we have a clear winner
        if sector_scores:
            best_sector = max(sector_scores.items(), key=lambda x: x[1])
            if best_sector[1] >= 2:  # At least 2 keyword matches
                return RoutingDecision(
                    query_type=QueryType.FINANCIAL,
                    sector=best_sector[0],
                    confidence=min(0.7 + (best_sector[1] * 0.1), 0.95),
                    reasoning=f"Matched {best_sector[1]} keywords for {best_sector[0]}",
                    should_process=True,
                    suggested_agent=f"{best_sector[0].value}Agent"
                )
        
        # Unclear - need LLM
        return None
    
    async def _llm_classify(self, query: str) -> RoutingDecision:
        """
        LLM-based classification for complex cases
        """
        prompt = f"""You are a financial query classifier. Analyze this query and determine:
1. Is it a FINANCIAL query or NON-FINANCIAL query?
2. If financial, which sector does it belong to?

Query: "{query}"

Available Sectors:
- IT (Software, Cloud, Tech Services)
- Pharma (Pharmaceuticals, Healthcare, Biotech)
- Banking (Banks, Financial Services, Fintech)
- Energy (Oil, Gas, Renewables)
- Automotive (Cars, EVs, Auto Industry)
- Consumer (FMCG, Retail, E-commerce)
- Manufacturing (Industrials, Heavy Industry)
- Real Estate (Property, REITs)
- Telecom (Telecom Services, Infrastructure)
- General (Cross-sector or unclear)

Classification Rules:
- If query is about stocks, companies, markets, finance → FINANCIAL
- If query is about recipes, cooking, travel, entertainment → NON-FINANCIAL
- If query mentions company names or financial terms → FINANCIAL
- Be strict: cooking pasta, movie recommendations, etc. are NON-FINANCIAL

Return ONLY a JSON object with this EXACT format:
{{
    "query_type": "financial" or "non_financial",
    "sector": "IT" or "Pharma" or "Banking" etc. (only if financial),
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON, no markdown, no extra text."""

        response = await self.llm_client.generate_response(prompt)
        
        # Parse JSON response
        import json
        try:
            # Clean response (remove markdown if present)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()
            
            data = json.loads(response)
            
            query_type = QueryType.FINANCIAL if data['query_type'].lower() == 'financial' else QueryType.NON_FINANCIAL
            
            if query_type == QueryType.NON_FINANCIAL:
                return RoutingDecision(
                    query_type=query_type,
                    confidence=data.get('confidence', 0.8),
                    reasoning=data.get('reasoning', 'LLM classified as non-financial'),
                    should_process=False,
                    rejection_message=self._generate_rejection_message(query)
                )
            else:
                sector_str = data.get('sector', 'General')
                sector = self._parse_sector(sector_str)
                
                return RoutingDecision(
                    query_type=query_type,
                    sector=sector,
                    confidence=data.get('confidence', 0.8),
                    reasoning=data.get('reasoning', 'LLM classified as financial'),
                    should_process=True,
                    suggested_agent=f"{sector.value}Agent"
                )
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            # Default to unclear
            return RoutingDecision(
                query_type=QueryType.UNCLEAR,
                confidence=0.3,
                reasoning="Failed to classify query",
                should_process=False,
                rejection_message="I'm having trouble understanding your query. Could you rephrase it?"
            )
    
    def _parse_sector(self, sector_str: str) -> Sector:
        """Parse sector string to Enum"""
        try:
            return Sector(sector_str)
        except:
            # Try to match
            sector_upper = sector_str.upper()
            for sector in Sector:
                if sector.value.upper() == sector_upper:
                    return sector
            return Sector.GENERAL
    
    def _generate_rejection_message(self, query: str) -> str:
        """Generate polite rejection message for non-financial queries"""
        messages = [
            "I'm a financial research assistant specialized in stock markets, companies, and financial analysis. I can't help with non-financial topics.",
            
            "I focus exclusively on financial research and market analysis. For other topics, I'd recommend using a general-purpose assistant.",
            
            "I'm designed to analyze stocks, companies, and financial markets. I'm not able to assist with non-financial queries.",
        ]
        
        # Pick based on query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["recipe", "cook", "food"]):
            return "I'm a financial research assistant, not a cooking expert! 👨‍🍳 I specialize in stock analysis, company research, and market insights. For recipes, I'd suggest a cooking website or general AI assistant."
        
        elif any(word in query_lower for word in ["movie", "film", "show"]):
            return "I'm focused on financial markets and company analysis, not entertainment! 🎬 For movie recommendations, try a general-purpose AI or IMDb."
        
        elif any(word in query_lower for word in ["travel", "vacation", "trip"]):
            return "I specialize in financial research, not travel planning! ✈️ For travel advice, check out travel websites or a general assistant."
        
        else:
            return messages[0] + "\n\n**I can help you with:**\n- Stock analysis\n- Company research\n- Market trends\n- Financial comparisons\n- Sector analysis"
    
    def get_available_sectors(self) -> list:
        """Get list of available sectors"""
        return [sector.value for sector in Sector]


# Sector Agent Base Class
class SectorAgent:
    """
    Base class for sector-specific agents
    Makes it easy to add new sectors
    """
    
    def __init__(self, sector: Sector, llm_client, search_client):
        self.sector = sector
        self.llm_client = llm_client
        self.search_client = search_client
        self.logger = logging.getLogger(f"{sector.value}Agent")
    
    def is_applicable(self, query: str) -> bool:
        """Check if this agent can handle the query"""
        raise NotImplementedError
    
    async def build_plan(self, query: str):
        """Build sector-specific research plan"""
        raise NotImplementedError
    
    async def run_research(self, plan):
        """Execute sector-specific research"""
        raise NotImplementedError


# Example: IT Sector Agent
class ITSectorAgent(SectorAgent):
    """Agent specialized in IT sector"""
    
    def __init__(self, llm_client, search_client):
        super().__init__(Sector.IT, llm_client, search_client)
    
    def is_applicable(self, query: str) -> bool:
        """Check if query is IT-related"""
        it_keywords = ["software", "cloud", "saas", "tech", "it services"]
        return any(kw in query.lower() for kw in it_keywords)
    
    async def build_plan(self, query: str):
        """Build IT-specific research plan"""
        from src.agents.research_planner import ResearchPlanner
        planner = ResearchPlanner(self.llm_client)
        
        # Add IT-specific context
        enhanced_query = f"{query} (Focus on: Cloud revenue, Digital transformation, AI/ML capabilities, Client retention)"
        
        return await planner.generate_plan(enhanced_query, sector="IT")
    
    async def run_research(self, plan):
        """Execute IT-specific research"""
        # This would use the deep executor with IT-specific metrics
        pass


# Example: Pharma Sector Agent
class PharmaSectorAgent(SectorAgent):
    """Agent specialized in Pharma sector"""
    
    def __init__(self, llm_client, search_client):
        super().__init__(Sector.PHARMA, llm_client, search_client)
    
    def is_applicable(self, query: str) -> bool:
        """Check if query is Pharma-related"""
        pharma_keywords = ["pharma", "drug", "medicine", "biotech", "clinical"]
        return any(kw in query.lower() for kw in pharma_keywords)
    
    async def build_plan(self, query: str):
        """Build Pharma-specific research plan"""
        from src.agents.research_planner import ResearchPlanner
        planner = ResearchPlanner(self.llm_client)
        
        # Add Pharma-specific context
        enhanced_query = f"{query} (Focus on: Drug pipeline, R&D spend, Patent expiries, FDA approvals, Biosimilars)"
        
        return await planner.generate_plan(enhanced_query, sector="Pharma")
    
    async def run_research(self, plan):
        """Execute Pharma-specific research"""
        pass


# Agent Registry
class AgentRegistry:
    """
    Registry of all sector agents
    Makes it easy to add new sectors
    """
    
    def __init__(self, llm_client, search_client):
        self.agents = {
            Sector.IT: ITSectorAgent(llm_client, search_client),
            Sector.PHARMA: PharmaSectorAgent(llm_client, search_client),
            # Add more sectors here
            # Sector.BANKING: BankingSectorAgent(llm_client, search_client),
            # Sector.ENERGY: EnergySectorAgent(llm_client, search_client),
            # etc.
        }
    
    def get_agent(self, sector: Sector):
        """Get agent for a sector"""
        return self.agents.get(sector)
    
    def add_agent(self, sector: Sector, agent: SectorAgent):
        """Add new sector agent"""
        self.agents[sector] = agent
    
    def list_sectors(self):
        """List available sectors"""
        return list(self.agents.keys())
