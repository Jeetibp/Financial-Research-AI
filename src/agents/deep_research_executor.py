"""
Deep Research Executor - Executes multi-step iterative research loops
Implements adaptive querying where each step informs the next
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from collections import defaultdict
from src.utils.financial_calculator import FinancialCalculator
from src.utils.number_extractor import NumberExtractor

logger = logging.getLogger(__name__)


class ResearchState(BaseModel):
    """Tracks state across research iterations"""
    plan_id: str
    current_step: int = 0
    total_steps: int
    completed_steps: List[int] = []
    step_results: Dict[int, Any] = {}
    discovered_topics: List[str] = []
    key_findings: List[str] = []
    pivot_history: List[Dict] = []
    status: str = "in_progress"  # "in_progress", "completed", "failed"


class StepResult(BaseModel):
    """Result from a single research step"""
    step_number: int
    action: str
    query_used: str
    data: Any
    insights: List[str] = []
    discovered_topics: List[str] = []
    suggested_next_queries: List[str] = []
    execution_time: float
    timestamp: str = ""


class DeepResearchExecutor:
    """
    Executes research plans with iterative, adaptive loops
    
    Key features:
    - Executes 5-20+ steps based on plan
    - Each step informs the next (adaptive)
    - Automatically spawns sub-loops for discovered topics
    - Tracks state across iterations
    - Supports streaming results
    """
    
    def __init__(self, research_agent, llm_client, search_client):
        self.research_agent = research_agent
        self.llm_client = llm_client
        self.search_client = search_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize calculation utilities
        self.calculator = FinancialCalculator()
        self.number_extractor = NumberExtractor()
        
        # Store active research sessions
        self.active_sessions: Dict[str, ResearchState] = {}
    
    async def execute_plan(
        self, 
        plan,
        callback=None  # Optional callback for streaming updates
    ) -> Dict:
        """
        Execute a research plan with iterative loops
        
        Args:
            plan: ResearchPlan object
            callback: Optional async function to stream updates
            
        Returns:
            Complete research results with synthesis
        """
        self.logger.info(f"Starting execution of plan {plan.plan_id}")
        
        # Initialize state
        state = ResearchState(
            plan_id=plan.plan_id,
            total_steps=plan.total_steps
        )
        self.active_sessions[plan.plan_id] = state
        
        # Track all results
        all_results = []
        context_accumulator = {}  # Accumulates context from previous steps
        
        try:
            # Execute each step in sequence
            for step in plan.steps:
                self.logger.info(f"Executing step {step.step_number}/{plan.total_steps}: {step.description}")
                
                # Check dependencies
                if step.depends_on:
                    await self._wait_for_dependencies(step.depends_on, state)
                
                # Execute the step with adaptive context
                result = await self._execute_step(
                    step, 
                    state, 
                    context_accumulator,
                    plan
                )
                
                # Store result
                all_results.append(result)
                state.step_results[step.step_number] = result
                state.completed_steps.append(step.step_number)
                state.current_step = step.step_number
                
                # Update accumulated context
                context_accumulator = self._update_context(
                    context_accumulator, 
                    result
                )
                
                # Check for discovered topics that need sub-loops
                if result.discovered_topics:
                    self.logger.info(f"Discovered topics: {result.discovered_topics}")
                    state.discovered_topics.extend(result.discovered_topics)
                    
                    # Decide if we should spawn sub-loops
                    should_expand = await self._should_expand_research(
                        result.discovered_topics,
                        state,
                        plan
                    )
                    
                    if should_expand:
                        sub_results = await self._spawn_sub_loop(
                            result.discovered_topics[0],  # Focus on first discovery
                            state,
                            context_accumulator,
                            plan
                        )
                        all_results.extend(sub_results)
                
                # Stream update if callback provided
                if callback:
                    await callback({
                        'type': 'step_completed',
                        'step': step.step_number,
                        'total': plan.total_steps,
                        'result': result.dict()
                    })
                
                # Adaptive pause between steps
                await asyncio.sleep(0.5)
            
            # Mark as completed
            state.status = "completed"
            
            # Synthesize final report
            final_report = await self._synthesize_results(
                plan,
                all_results,
                state
            )
            
            return {
                'success': True,
                'plan_id': plan.plan_id,
                'total_steps_executed': len(all_results),
                'discovered_topics': state.discovered_topics,
                'key_findings': state.key_findings,
                'results': [r.dict() for r in all_results],
                'final_report': final_report,
                'execution_time': sum(r.execution_time for r in all_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing plan: {e}")
            state.status = "failed"
            raise
        
        finally:
            # Cleanup
            if plan.plan_id in self.active_sessions:
                del self.active_sessions[plan.plan_id]
    
    async def _execute_step(
        self, 
        step, 
        state: ResearchState,
        context: Dict,
        plan
    ) -> StepResult:
        """
        Execute a single research step
        
        This is where the ADAPTIVE magic happens - we modify the query
        based on previous findings
        """
        import time
        start_time = time.time()
        
        # ADAPTIVE QUERY GENERATION
        # Instead of using the query as-is, we adapt it based on context
        adapted_query = await self._adapt_query(
            step.query,
            context,
            state.discovered_topics,
            plan
        )
        
        self.logger.info(f"Original query: {step.query}")
        self.logger.info(f"Adapted query: {adapted_query}")
        
        # Execute based on action type
        if step.action == "web_search":
            data = await self._execute_web_search(adapted_query)
        
        elif step.action == "rag_query":
            data = await self._execute_rag_query(adapted_query)
        
        elif step.action == "api_call":
            data = await self._execute_api_call(adapted_query)
        
        elif step.action == "calculate":
            data = await self._execute_calculation(adapted_query, context)
        
        else:
            raise ValueError(f"Unknown action type: {step.action}")
        
        # Extract insights from the result
        insights = await self._extract_insights(data, adapted_query)
        
        # Detect new topics to explore
        discovered_topics = await self._detect_topics(data, insights)
        
        # Generate suggested next queries (adaptive)
        suggested_queries = await self._generate_next_queries(
            data, 
            insights, 
            discovered_topics,
            plan
        )
        
        execution_time = time.time() - start_time
        
        return StepResult(
            step_number=step.step_number,
            action=step.action,
            query_used=adapted_query,
            data=data,
            insights=insights,
            discovered_topics=discovered_topics,
            suggested_next_queries=suggested_queries,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    async def _adapt_query(
        self, 
        original_query: str,
        context: Dict,
        discovered_topics: List[str],
        plan
    ) -> str:
        """
        CRITICAL FUNCTION: Adapts queries based on previous findings
        
        This is what makes it "iterative" and "deep"
        """
        
        # If no context yet, use original query
        if not context:
            return original_query
        
        # Build context summary
        context_summary = self._build_context_summary(context)
        
        # Generate adapted query using LLM
        prompt = f"""You are adapting a research query based on previous findings.

Original Query: {original_query}

Previous Findings:
{context_summary}

Recently Discovered Topics: {', '.join(discovered_topics) if discovered_topics else 'None yet'}

Research Goal: {plan.query}
Sector: {plan.sector}

Task: Adapt the original query to be MORE SPECIFIC based on what we've learned.

Examples:
- If we found "AI revenue growing 45%", adapt "get revenue" to "get AI segment revenue breakdown"
- If we discovered "biosimilars" trend, adapt generic pharma query to focus on biosimilars
- If we found a merger, adapt to search for "financial impact of [specific merger]"

Return ONLY the adapted query, nothing else."""

        adapted = await self.llm_client.generate_response(prompt)
        return adapted.strip()
    
    async def _execute_web_search(self, query: str) -> Dict:
        """Execute web search"""
        try:
            results = await self.search_client.search(query, max_results=5)
            return {
                'source': 'web_search',
                'query': query,
                'results': results
            }
        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return {'source': 'web_search', 'error': str(e)}
    
    async def _execute_rag_query(self, query: str) -> Dict:
        """Execute RAG query on uploaded documents"""
        # This would call your existing RAG system
        try:
            # Placeholder - replace with actual RAG call
            return {
                'source': 'rag',
                'query': query,
                'results': 'RAG results here'
            }
        except Exception as e:
            self.logger.error(f"RAG query error: {e}")
            return {'source': 'rag', 'error': str(e)}
    
    async def _execute_api_call(self, query: str) -> Dict:
        """Execute financial API call"""
        # Call your financial APIs (yfinance, Alpha Vantage, etc.)
        try:
            # Placeholder - replace with actual API calls
            return {
                'source': 'api',
                'query': query,
                'results': 'API results here'
            }
        except Exception as e:
            self.logger.error(f"API call error: {e}")
            return {'source': 'api', 'error': str(e)}
    
    async def _execute_calculation(self, query: str, context: Dict) -> Dict:
        """
        Execute mathematical calculation
        This will be expanded in Gap 4 (Math Guardrail)
        """
        # Placeholder for now - will be implemented in Gap 4
        return {
            'source': 'calculation',
            'query': query,
            'results': 'Calculation results here'
        }
    
    async def _extract_insights(self, data: Dict, query: str) -> List[str]:
        """Extract key insights from step results"""
        prompt = f"""Extract 2-3 key insights from this research data.

Query: {query}
Data: {str(data)[:2000]}

Return a JSON list of strings, each being one concise insight.
Focus on:
- Surprising findings
- Trends or patterns
- Important metrics or changes
- Competitive dynamics

Example: ["Revenue grew 35% YoY", "Cloud segment now 60% of total revenue"]

Return ONLY the JSON array."""

        response = await self.llm_client.generate_response(prompt)
        
        import json
        try:
            insights = json.loads(response)
            return insights if isinstance(insights, list) else []
        except:
            return []
    
    async def _detect_topics(self, data: Dict, insights: List[str]) -> List[str]:
        """
        Detect new topics worth exploring deeper
        
        Example: If we see "biosimilars" mentioned multiple times,
        flag it as a discovered topic
        """
        prompt = f"""Analyze this data and identify 1-2 specific topics that deserve deeper research.

Data: {str(data)[:1500]}
Insights: {insights}

Look for:
- Emerging trends (e.g., "biosimilars", "GenAI services")
- Specific product lines or segments
- Market dynamics or shifts
- Competitive threats or opportunities

Return a JSON list of specific topic strings worth investigating.
Return ONLY topics that are SPECIFIC and ACTIONABLE.

Example: ["biosimilars market", "GenAI consulting services"]

Return ONLY the JSON array, or empty array [] if no significant topics found."""

        response = await self.llm_client.generate_response(prompt)
        
        import json
        try:
            topics = json.loads(response)
            return topics if isinstance(topics, list) else []
        except:
            return []
    
    async def _generate_next_queries(
        self,
        data: Dict,
        insights: List[str],
        discovered_topics: List[str],
        plan
    ) -> List[str]:
        """Generate suggested queries for next steps (adaptive)"""
        
        if not discovered_topics:
            return []
        
        prompt = f"""Based on these discoveries, suggest 2-3 follow-up research queries.

Sector: {plan.sector}
Discovered Topics: {discovered_topics}
Current Insights: {insights}

Generate specific, actionable queries to explore these topics deeper.

Examples:
- "biosimilars market size and growth rate 2024-2026"
- "GenAI revenue breakdown by major IT companies"
- "impact of [specific merger] on market share"

Return a JSON array of query strings.
Return ONLY the JSON array."""

        response = await self.llm_client.generate_response(prompt)
        
        import json
        try:
            queries = json.loads(response)
            return queries if isinstance(queries, list) else []
        except:
            return []
    
    async def _should_expand_research(
        self,
        topics: List[str],
        state: ResearchState,
        plan
    ) -> bool:
        """
        Decide if we should spawn a sub-loop for discovered topics
        
        Criteria:
        - Topic is highly relevant
        - We have capacity (not too many steps already)
        - Topic hasn't been explored yet
        """
        if not topics:
            return False
        
        # Check if we have capacity for more steps
        if len(state.completed_steps) >= 20:  # Max 20 steps total
            return False
        
        # Check if topic is already being explored
        if topics[0] in state.discovered_topics[:-1]:  # Already seen
            return False
        
        # For now, expand if we're in "deep" mode
        if plan.estimated_depth == "deep":
            return True
        
        return False
    
    async def _spawn_sub_loop(
        self,
        topic: str,
        state: ResearchState,
        context: Dict,
        plan
    ) -> List[StepResult]:
        """
        Spawn a mini research loop for a discovered topic
        
        This creates 3-5 additional steps focused on the new topic
        """
        self.logger.info(f"🔄 Spawning sub-loop for topic: {topic}")
        
        # Record the pivot
        state.pivot_history.append({
            'topic': topic,
            'at_step': state.current_step,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate sub-queries for this topic
        sub_queries = await self._generate_sub_queries(topic, plan)
        
        # Execute each sub-query
        sub_results = []
        for i, query in enumerate(sub_queries[:3], 1):  # Limit to 3 sub-steps
            self.logger.info(f"  Sub-step {i}: {query}")
            
            # Create a synthetic step
            from src.agents.research_planner import ResearchStep
            sub_step = ResearchStep(
                step_number=state.current_step + i,
                action="web_search",
                description=f"Deep dive on {topic}",
                tool="tavily",
                query=query,
                expected_output=f"Detailed info on {topic}"
            )
            
            # Execute it
            result = await self._execute_step(sub_step, state, context, plan)
            sub_results.append(result)
            
            # Update state
            state.step_results[sub_step.step_number] = result
            state.current_step = sub_step.step_number
        
        self.logger.info(f"✅ Sub-loop completed with {len(sub_results)} steps")
        return sub_results
    
    async def _generate_sub_queries(self, topic: str, plan) -> List[str]:
        """Generate focused queries for a discovered topic"""
        prompt = f"""Generate 3 specific research queries to deeply explore this topic.

Topic: {topic}
Sector: {plan.sector}
Original Research Goal: {plan.query}

Generate queries that explore:
1. Market size, growth, and trends
2. Key players and competitive landscape
3. Future outlook and opportunities

Return a JSON array of 3 query strings.
Make them specific and actionable.

Return ONLY the JSON array."""

        response = await self.llm_client.generate_response(prompt)
        
        import json
        try:
            queries = json.loads(response)
            return queries if isinstance(queries, list) else []
        except:
            return [
                f"{topic} market size and growth",
                f"{topic} key players and competition",
                f"{topic} trends and outlook"
            ]
    
    def _update_context(self, context: Dict, result: StepResult) -> Dict:
        """Update accumulated context with new findings"""
        context[f"step_{result.step_number}"] = {
            'action': result.action,
            'query': result.query_used,
            'insights': result.insights,
            'topics': result.discovered_topics
        }
        return context
    
    def _build_context_summary(self, context: Dict) -> str:
        """Build a summary of previous findings"""
        summary_parts = []
        for step_key, step_data in context.items():
            if step_data.get('insights'):
                summary_parts.append(f"- {', '.join(step_data['insights'])}")
        
        return '\n'.join(summary_parts[-5:])  # Last 5 steps only
    
    async def _wait_for_dependencies(self, dependencies: List[int], state: ResearchState):
        """Wait for dependent steps to complete"""
        for dep in dependencies:
            while dep not in state.completed_steps:
                await asyncio.sleep(0.1)
    
    async def _execute_calculation(self, query: str, context: Dict) -> Dict:
        """
        Execute mathematical calculation - NO LLM MATH
        
        This extracts numbers from context and calculates programmatically
        """
        self.logger.info(f"Executing calculation: {query}")
        
        # Step 1: Parse what calculation is needed
        calc_type = await self._identify_calculation_type(query)
        
        # Step 2: Extract required numbers from context
        numbers = self._extract_numbers_from_context(context, calc_type)
        
        # Step 3: Execute calculation in pure Python
        result = self._perform_calculation(calc_type, numbers)
        
        return {
            'source': 'calculation',
            'query': query,
            'calculation_type': calc_type,
            'inputs': numbers,
            'result': result.dict() if result else None,
            'formula': result.formula if result else None
        }
    
    async def _identify_calculation_type(self, query: str) -> str:
        """Identify what type of calculation is needed"""
        query_lower = query.lower()
        
        if 'growth' in query_lower or 'yoy' in query_lower or 'qoq' in query_lower:
            return 'growth_rate'
        elif 'cagr' in query_lower:
            return 'cagr'
        elif 'margin' in query_lower:
            if 'gross' in query_lower:
                return 'gross_margin'
            elif 'ebitda' in query_lower:
                return 'ebitda_margin'
            elif 'operating' in query_lower:
                return 'operating_margin'
            else:
                return 'net_margin'
        elif 'roe' in query_lower or 'return on equity' in query_lower:
            return 'roe'
        elif 'roa' in query_lower or 'return on assets' in query_lower:
            return 'roa'
        elif 'p/e' in query_lower or 'pe ratio' in query_lower or 'price to earnings' in query_lower:
            return 'pe_ratio'
        elif 'debt' in query_lower and 'equity' in query_lower:
            return 'debt_to_equity'
        elif 'current ratio' in query_lower:
            return 'current_ratio'
        elif 'market cap' in query_lower:
            return 'market_cap'
        else:
            return 'general'
    
    def _extract_numbers_from_context(self, context: Dict, calc_type: str) -> Dict[str, float]:
        """
        Extract numbers needed for calculation from previous step results
        
        This is the KEY function - it pulls numbers from data, not from LLM
        """
        numbers = {}
        
        # Collect all data from previous steps
        all_text = ""
        for step_key, step_data in context.items():
            if isinstance(step_data, dict) and 'data' in step_data:
                all_text += str(step_data['data']) + " "
        
        # Use number extractor
        metrics = self.number_extractor.extract_financial_metrics({'text': all_text})
        
        # Map to required inputs based on calc_type
        if calc_type == 'growth_rate':
            # Need current and previous values
            all_numbers = self.number_extractor.extract_all_numbers(all_text)
            if len(all_numbers) >= 2:
                numbers['current_value'] = all_numbers[-1][0]  # Most recent
                numbers['previous_value'] = all_numbers[-2][0]  # Previous
        
        elif calc_type in ['net_margin', 'gross_margin', 'operating_margin', 'ebitda_margin']:
            numbers['profit'] = metrics.get('profit', 0)
            numbers['revenue'] = metrics.get('revenue', 0)
        
        elif calc_type == 'pe_ratio':
            numbers['stock_price'] = metrics.get('stock_price', 0)
            numbers['eps'] = metrics.get('eps', 0)
        
        # Add more mappings as needed
        
        return numbers
    
    def _perform_calculation(self, calc_type: str, numbers: Dict[str, float]):
        """
        Perform the actual calculation using FinancialCalculator
        
        NO LLM INVOLVED - Pure Python math
        """
        try:
            if calc_type == 'growth_rate':
                return self.calculator.calculate_growth_rate(
                    current_value=numbers.get('current_value', 0),
                    previous_value=numbers.get('previous_value', 0)
                )
            
            elif calc_type in ['net_margin', 'gross_margin', 'operating_margin', 'ebitda_margin']:
                margin_type = calc_type.replace('_margin', '')
                return self.calculator.calculate_margin(
                    profit=numbers.get('profit', 0),
                    revenue=numbers.get('revenue', 0),
                    margin_type=margin_type
                )
            
            elif calc_type == 'roe':
                return self.calculator.calculate_roe(
                    net_income=numbers.get('net_income', 0),
                    shareholders_equity=numbers.get('equity', 0)
                )
            
            elif calc_type == 'roa':
                return self.calculator.calculate_roa(
                    net_income=numbers.get('net_income', 0),
                    total_assets=numbers.get('assets', 0)
                )
            
            elif calc_type == 'pe_ratio':
                return self.calculator.calculate_pe_ratio(
                    stock_price=numbers.get('stock_price', 0),
                    eps=numbers.get('eps', 0)
                )
            
            elif calc_type == 'debt_to_equity':
                return self.calculator.calculate_debt_to_equity(
                    total_debt=numbers.get('debt', 0),
                    shareholders_equity=numbers.get('equity', 0)
                )
            
            elif calc_type == 'current_ratio':
                return self.calculator.calculate_current_ratio(
                    current_assets=numbers.get('current_assets', 0),
                    current_liabilities=numbers.get('current_liabilities', 0)
                )
            
            elif calc_type == 'cagr':
                return self.calculator.calculate_cagr(
                    starting_value=numbers.get('starting_value', 0),
                    ending_value=numbers.get('ending_value', 0),
                    num_years=numbers.get('years', 1)
                )
            
            else:
                self.logger.warning(f"Unknown calculation type: {calc_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Calculation error: {e}")
            return None
    
    async def _synthesize_results(
        self,
        plan,
        results: List[StepResult],
        state: ResearchState
    ) -> str:
        """Synthesize all results into final report"""
        
        # Collect all insights
        all_insights = []
        for result in results:
            all_insights.extend(result.insights)
        
        # Collect all data
        all_data = [r.data for r in results]
        
        # Use LLM to synthesize
        prompt = f"""Synthesize this deep research into a comprehensive report.

Research Query: {plan.query}
Sector: {plan.sector}
Total Steps Executed: {len(results)}
Discovered Topics: {state.discovered_topics}

Key Insights:
{chr(10).join(f'- {insight}' for insight in all_insights)}

Task: Write a comprehensive financial research report with these sections:
1. Executive Summary
2. Key Findings
3. Financial Analysis
4. Market Position & Competitive Landscape
5. Risks & Opportunities
6. Investment Outlook

Make it factual, data-driven, and actionable.
Use markdown formatting.
"""

        report = await self.llm_client.generate_response(prompt)
        return report
