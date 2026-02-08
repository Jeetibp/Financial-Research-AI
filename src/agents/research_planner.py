"""
Research Planner - Generates detailed research plans before execution
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchStep(BaseModel):
    """Single step in research plan"""
    step_number: int
    action: str  # "web_search", "rag_query", "api_call", "calculate"
    description: str
    tool: str
    query: str
    expected_output: str
    depends_on: Optional[List[int]] = None  # Steps this depends on


class ResearchPlan(BaseModel):
    """Complete research plan"""
    plan_id: str
    query: str
    sector: str
    analysis_type: str  # "company", "sector", "comparative"
    total_steps: int
    estimated_depth: str  # "shallow", "medium", "deep"
    steps: List[ResearchStep]
    data_sources: List[str]
    key_metrics: List[str]
    expected_outputs: List[str]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending_approval"  # "pending_approval", "approved", "rejected", "completed"


class ResearchPlanner:
    """Generates intelligent research plans based on queries"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def generate_plan(
        self, 
        query: str, 
        sector: Optional[str] = None,
        analysis_type: Optional[str] = None
    ) -> ResearchPlan:
        """
        Generate a detailed research plan for the query
        
        Args:
            query: User's research question
            sector: IT, Pharma, Banking, etc.
            analysis_type: company, sector, or comparative
            
        Returns:
            ResearchPlan with steps, tools, and expected outputs
        """
        self.logger.info(f"Generating research plan for: {query}")
        
        # Detect sector if not provided
        if not sector:
            sector = await self._detect_sector(query)
        
        # Detect analysis type if not provided
        if not analysis_type:
            analysis_type = await self._detect_analysis_type(query)
        
        # Determine research depth
        depth = await self._determine_depth(query, sector)
        
        # Generate steps using LLM
        steps = await self._generate_research_steps(query, sector, analysis_type, depth)
        
        # Create plan
        plan = ResearchPlan(
            plan_id=self._generate_plan_id(),
            query=query,
            sector=sector,
            analysis_type=analysis_type,
            total_steps=len(steps),
            estimated_depth=depth,
            steps=steps,
            data_sources=self._identify_data_sources(steps),
            key_metrics=self._identify_key_metrics(sector, analysis_type),
            expected_outputs=self._define_expected_outputs(analysis_type)
        )
        
        self.logger.info(f"Plan generated with {len(steps)} steps")
        return plan
    
    async def _detect_sector(self, query: str) -> str:
        """Detect which sector this query belongs to"""
        prompt = f"""Analyze this financial research query and identify the primary sector.

Query: {query}

Available sectors:
- IT (Information Technology, Software, Cloud)
- Pharma (Pharmaceuticals, Healthcare, Biotech)
- Banking (Banks, Financial Services, Fintech)
- Energy (Oil, Gas, Renewables)
- Manufacturing (Industrials, Auto)
- Consumer (Retail, FMCG, E-commerce)
- General (Cross-sector or unclear)

Return ONLY the sector name, nothing else."""

        response = await self.llm_client.generate_response(prompt)
        sector = response.strip()
        
        valid_sectors = ["IT", "Pharma", "Banking", "Energy", "Manufacturing", "Consumer", "General"]
        return sector if sector in valid_sectors else "General"
    
    async def _detect_analysis_type(self, query: str) -> str:
        """Detect type of analysis needed"""
        query_lower = query.lower()
        
        # Comparative keywords
        if any(word in query_lower for word in ["vs", "versus", "compare", "comparison", "difference"]):
            return "comparative"
        
        # Company-specific keywords
        if any(word in query_lower for word in ["company", "stock", "ticker", "share price"]):
            return "company"
        
        # Sector keywords
        if any(word in query_lower for word in ["sector", "industry", "market", "trend"]):
            return "sector"
        
        # Default to company if specific company names detected
        # This could be enhanced with NER
        return "company"
    
    async def _determine_depth(self, query: str, sector: str) -> str:
        """Determine how deep the research should go"""
        # Simple heuristics - can be enhanced
        query_lower = query.lower()
        
        # Deep research indicators
        deep_keywords = ["deep dive", "comprehensive", "detailed analysis", "full report", "in-depth"]
        if any(kw in query_lower for kw in deep_keywords):
            return "deep"  # 15-20 steps
        
        # Shallow research indicators
        shallow_keywords = ["quick", "brief", "summary", "overview", "current price"]
        if any(kw in query_lower for kw in shallow_keywords):
            return "shallow"  # 3-5 steps
        
        # Medium by default
        return "medium"  # 8-12 steps
    
    async def _generate_research_steps(
        self, 
        query: str, 
        sector: str, 
        analysis_type: str,
        depth: str
    ) -> List[ResearchStep]:
        """Generate detailed research steps"""
        
        # Map depth to step counts
        step_counts = {
            "shallow": (3, 5),
            "medium": (8, 12),
            "deep": (15, 20)
        }
        min_steps, max_steps = step_counts[depth]
        
        # Generate steps based on analysis type and sector
        if analysis_type == "company":
            steps = await self._generate_company_steps(query, sector, min_steps, max_steps)
        elif analysis_type == "comparative":
            steps = await self._generate_comparative_steps(query, sector, min_steps, max_steps)
        else:  # sector analysis
            steps = await self._generate_sector_steps(query, sector, min_steps, max_steps)
        
        return steps
    
    async def _generate_company_steps(
        self, 
        query: str, 
        sector: str,
        min_steps: int,
        max_steps: int
    ) -> List[ResearchStep]:
        """Generate steps for company analysis"""
        
        # Use LLM to generate intelligent, adaptive steps
        prompt = f"""You are a financial research planner. Generate {min_steps}-{max_steps} research steps for this query.

Query: {query}
Sector: {sector}
Analysis Type: Company Analysis

Generate a JSON list of research steps. Each step should have:
- step_number: integer
- action: "web_search", "rag_query", "api_call", or "calculate"
- description: what this step does
- tool: specific tool to use (tavily, serpapi, pdf_rag, yfinance, python_calc)
- query: the specific query/search term
- expected_output: what data we expect to get
- depends_on: list of step numbers this depends on (or null)

Make steps ADAPTIVE - each step should build on previous results.
Example: Step 2 might search for trends discovered in Step 1.

Important rules:
1. Start with basic info gathering (company overview, stock price)
2. Then financial metrics (revenue, profit, margins)
3. Use CALCULATE action for all math (growth rates, ratios)
4. Then competitive analysis and market position
5. Finally risks, catalysts, and outlook
6. Make later steps depend on discoveries in earlier steps

Return ONLY valid JSON array, no markdown."""

        response = await self.llm_client.generate_response(prompt)
        
        # Parse JSON response
        import json
        try:
            steps_data = json.loads(response)
            steps = [ResearchStep(**step) for step in steps_data]
            return steps
        except Exception as e:
            self.logger.error(f"Error parsing steps: {e}")
            # Fallback to default steps
            return self._get_default_company_steps()
    
    def _get_default_company_steps(self) -> List[ResearchStep]:
        """Fallback default steps for company analysis"""
        return [
            ResearchStep(
                step_number=1,
                action="web_search",
                description="Get current stock price and basic company info",
                tool="tavily",
                query="[company] current stock price market cap",
                expected_output="Stock price, market cap, basic company details"
            ),
            ResearchStep(
                step_number=2,
                action="rag_query",
                description="Extract latest financial results from documents",
                tool="pdf_rag",
                query="Latest quarterly revenue, profit, and key metrics",
                expected_output="Q revenue, net income, EBITDA, margins"
            ),
            ResearchStep(
                step_number=3,
                action="calculate",
                description="Calculate year-over-year growth rates",
                tool="python_calc",
                query="Calculate revenue growth, profit growth",
                expected_output="YoY growth percentages",
                depends_on=[2]
            ),
            ResearchStep(
                step_number=4,
                action="web_search",
                description="Research recent news and developments",
                tool="tavily",
                query="[company] recent news announcements last 30 days",
                expected_output="Recent news, product launches, partnerships"
            ),
            ResearchStep(
                step_number=5,
                action="web_search",
                description="Analyze competitive positioning",
                tool="serpapi",
                query="[company] market share competitors analysis",
                expected_output="Market share, key competitors, positioning"
            )
        ]
    
    async def _generate_comparative_steps(
        self, 
        query: str, 
        sector: str,
        min_steps: int,
        max_steps: int
    ) -> List[ResearchStep]:
        """Generate steps for comparative analysis"""
        # Similar structure to company steps but comparing multiple entities
        return self._get_default_company_steps()  # Simplified for now
    
    async def _generate_sector_steps(
        self, 
        query: str, 
        sector: str,
        min_steps: int,
        max_steps: int
    ) -> List[ResearchStep]:
        """Generate steps for sector analysis"""
        # Similar structure but focused on sector-wide trends
        return self._get_default_company_steps()  # Simplified for now
    
    def _identify_data_sources(self, steps: List[ResearchStep]) -> List[str]:
        """Extract unique data sources from steps"""
        sources = set()
        for step in steps:
            sources.add(step.tool)
        return sorted(list(sources))
    
    def _identify_key_metrics(self, sector: str, analysis_type: str) -> List[str]:
        """Define key metrics based on sector"""
        base_metrics = ["Revenue", "Net Income", "EBITDA", "Margins"]
        
        sector_metrics = {
            "IT": ["ARR", "Customer Retention", "Cloud Revenue %"],
            "Pharma": ["R&D Spend", "Pipeline Drugs", "Patent Expiries"],
            "Banking": ["NIM", "NPAs", "CAR"],
        }
        
        return base_metrics + sector_metrics.get(sector, [])
    
    def _define_expected_outputs(self, analysis_type: str) -> List[str]:
        """Define what the final report should contain"""
        if analysis_type == "company":
            return [
                "Executive Summary",
                "Financial Performance Table",
                "Growth Analysis",
                "Competitive Position",
                "Risks & Catalysts",
                "Investment Outlook"
            ]
        elif analysis_type == "comparative":
            return [
                "Executive Summary",
                "Side-by-Side Comparison Table",
                "Relative Strengths & Weaknesses",
                "Recommendation"
            ]
        else:  # sector
            return [
                "Executive Summary",
                "Market Size & Growth",
                "Key Trends",
                "Major Players",
                "Outlook & Opportunities"
            ]
    
    def _generate_plan_id(self) -> str:
        """Generate unique plan ID"""
        import uuid
        return f"plan_{uuid.uuid4().hex[:12]}"
