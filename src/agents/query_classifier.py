"""
Intelligent Query Classifier - Tiered Complexity Detection
Classifies queries into INSTANT, STANDARD, or DEEP research modes
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryClassification:
    """Result of query classification"""
    tier: str  # INSTANT, STANDARD, DEEP
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_steps: int
    auto_execute: bool
    requires_approval: bool
    estimated_time: str

class IntelligentQueryClassifier:
    """
    Advanced query classifier that understands context and intent
    to avoid over-classification of simple queries
    """
    
    # Pattern definitions
    INSTANT_PATTERNS = [
        # Direct price queries
        r'(what is|what\'s|current|latest|today\'?s?)\s+(price|stock|value|quote)',
        r'(show|get|find|tell)\s+(me\s+)?(the\s+)?(price|stock|quote)',
        r'(price|stock)\s+of\s+\w+(\s+and\s+\w+)?$',
        r'^\w+\s+(price|stock)$',
        
        # Follow-up references
        r'^(both|these|them|those)(\s+companies?)?$',
        r'^(current|latest)\s+(price|stock)\s+of\s+(both|these|them)',
        
        # Simple factual
        r'^(who is|what is|when did|where is)',
        r'^(ceo|headquarters|founded|sector|industry)\s+of',
    ]
    
    STANDARD_PATTERNS = [
        # Comparisons
        r'compare\s+\w+\s+(vs|versus|and)\s+\w+',
        r'\w+\s+(vs|versus)\s+\w+',
        
        # Performance queries
        r'(performance|growth|trend|revenue)\s+of',
        r'(analyze|analysis)\s+\w+',
        
        # Recent data
        r'(recent|latest|last\s+(quarter|year|month))',
        r'(q[1-4]|fy)\s*\d{4}',
    ]
    
    DEEP_PATTERNS = [
        # Explicit deep requests
        r'(deep|comprehensive|detailed|thorough|extensive)\s+(analysis|research|dive|study)',
        r'(analyze|evaluate|assess|investigate)\s+.{30,}',  # Long queries
        
        # Forecasting
        r'(forecast|predict|project|outlook|future)',
        
        # Multi-factor analysis
        r'.*(and|along with|including|as well as).{20,}',  # Multiple aspects
        
        # Sector-wide
        r'(sector|industry|market)\s+(analysis|trends|dynamics|landscape)',
    ]
    
    FOLLOW_UP_INDICATORS = [
        'both', 'these', 'them', 'those', 'the two', 'the companies',
        'they', 'their', 'same companies', 'above companies'
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def classify(
        self, 
        query: str, 
        conversation_context: Optional[Dict] = None
    ) -> QueryClassification:
        """
        Classify query into appropriate tier
        
        Args:
            query: User's query
            conversation_context: Previous conversation data
            
        Returns:
            QueryClassification with tier and metadata
        """
        query_lower = query.lower().strip()
        
        # Check for instant patterns first (highest priority)
        if self._is_instant_query(query_lower, conversation_context):
            return QueryClassification(
                tier='INSTANT',
                confidence=0.95,
                reasoning='Direct factual query - immediate answer available',
                suggested_steps=1,
                auto_execute=True,
                requires_approval=False,
                estimated_time='<2s'
            )
        
        # Multi-company comparison = DEEP tier (requires comprehensive analysis)
        comparison_keywords = ['compare', 'vs', 'versus', 'difference between', 'vs.', 'comparison']
        has_comparison = any(kw in query_lower for kw in comparison_keywords)
        company_count = self._count_companies_in_query(query_lower)
        
        if has_comparison and company_count >= 2:
            return QueryClassification(
                tier='DEEP',
                confidence=0.95,
                reasoning='Multi-company comparison requires comprehensive analysis',
                suggested_steps=15,
                auto_execute=False,
                requires_approval=True,
                estimated_time='30-50s'
            )
        
        # Check for deep patterns
        if self._is_deep_query(query_lower):
            return QueryClassification(
                tier='DEEP',
                confidence=0.85,
                reasoning='Complex multi-factor analysis required',
                suggested_steps=12,
                auto_execute=False,
                requires_approval=True,
                estimated_time='25-40s'
            )
        
        # Default to standard for analytical queries
        if self._is_standard_query(query_lower):
            return QueryClassification(
                tier='STANDARD',
                confidence=0.90,
                reasoning='Moderate analysis with multiple data points',
                suggested_steps=5,
                auto_execute=True,
                requires_approval=False,
                estimated_time='8-15s'
            )
        
        # Fallback based on query length and complexity
        complexity_score = self._calculate_complexity_score(query)
        
        if complexity_score < 3:
            tier = 'INSTANT'
            steps = 1
            auto = True
            approval = False
            time_est = '<2s'
        elif complexity_score < 6:
            tier = 'STANDARD'
            steps = 5
            auto = True
            approval = False
            time_est = '8-15s'
        else:
            tier = 'DEEP'
            steps = 12
            auto = False
            approval = True
            time_est = '25-40s'
        
        return QueryClassification(
            tier=tier,
            confidence=0.70,
            reasoning=f'Complexity score: {complexity_score}/10',
            suggested_steps=steps,
            auto_execute=auto,
            requires_approval=approval,
            estimated_time=time_est
        )
    
    def _is_instant_query(self, query: str, context: Optional[Dict]) -> bool:
        """Check if query matches instant patterns"""
        
        # Pattern matching
        for pattern in self.INSTANT_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.info(f"✅ Instant pattern matched: {pattern}")
                return True
        
        # Context-aware follow-up detection
        if context and self._is_follow_up(query, context):
            self.logger.info(f"✅ Follow-up query detected with context")
            return True
        
        # Very short queries with known entities
        if len(query.split()) <= 4 and self._has_known_entity(query):
            self.logger.info(f"✅ Short query with known entity")
            return True
        
        return False
    
    def _is_standard_query(self, query: str) -> bool:
        """Check if query matches standard patterns"""
        for pattern in self.STANDARD_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.info(f"📊 Standard pattern matched: {pattern}")
                return True
        return False
    
    def _is_deep_query(self, query: str) -> bool:
        """Check if query matches deep research patterns"""
        for pattern in self.DEEP_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                self.logger.info(f"🔬 Deep pattern matched: {pattern}")
                return True
        return False
    
    def _count_companies_in_query(self, query: str) -> int:
        """Count number of companies mentioned in query"""
        # Common company names and tickers
        known_companies = [
            'microsoft', 'msft', 'google', 'googl', 'alphabet', 'goog',
            'apple', 'aapl', 'amazon', 'amzn', 'tesla', 'tsla',
            'meta', 'facebook', 'fb', 'netflix', 'nflx', 'nvidia', 'nvda',
            'intel', 'intc', 'amd', 'ibm', 'oracle', 'orcl', 'salesforce', 'crm',
            'adobe', 'adbe', 'cisco', 'csco', 'paypal', 'pypl', 'walmart', 'wmt',
            'visa', 'v', 'mastercard', 'ma', 'jpmorgan', 'jpm', 'boeing', 'ba',
            'disney', 'dis', 'coca-cola', 'ko', 'pepsi', 'pep', 'starbucks', 'sbux'
        ]
        
        count = 0
        query_words = query.lower().split()
        for company in known_companies:
            if company in query_words or company in query.lower():
                count += 1
        
        # Also check for 'and' between potential company names
        if ' and ' in query.lower() or ' & ' in query:
            count = max(count, 2)
        
        return count
    
    def _is_follow_up(self, query: str, context: Dict) -> bool:
        """Detect if query is a follow-up using context"""
        if not context:
            return False
        
        # Check for pronoun references
        has_pronoun = any(
            indicator in query.lower() 
            for indicator in self.FOLLOW_UP_INDICATORS
        )
        
        # Check recency (within 5 minutes)
        last_query_time = context.get('last_query_time')
        if last_query_time:
            time_diff = (datetime.now() - last_query_time).total_seconds()
            is_recent = time_diff < 300  # 5 minutes
        else:
            is_recent = False
        
        # Check if entities from previous query exist
        has_entities = bool(context.get('companies') or context.get('tickers'))
        
        return has_pronoun and is_recent and has_entities
    
    def _has_known_entity(self, query: str) -> bool:
        """Check if query contains recognized companies/tickers"""
        known_entities = [
            'microsoft', 'msft', 'google', 'googl', 'alphabet',
            'apple', 'aapl', 'amazon', 'amzn', 'tesla', 'tsla',
            'meta', 'facebook', 'netflix', 'nflx', 'nvidia', 'nvda'
        ]
        return any(entity in query.lower() for entity in known_entities)
    
    def _calculate_complexity_score(self, query: str) -> int:
        """Calculate complexity score 0-10"""
        score = 0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 15:
            score += 3
        elif word_count > 8:
            score += 2
        elif word_count > 4:
            score += 1
        
        # Multiple questions
        if query.count('?') > 1:
            score += 2
        
        # Conjunctions (multiple aspects)
        conjunctions = query.lower().count(' and ') + query.lower().count(' or ')
        score += min(conjunctions, 2)
        
        # Specific analytical terms
        analytical_terms = [
            'analyze', 'compare', 'evaluate', 'assess',
            'forecast', 'predict', 'trend', 'impact'
        ]
        score += sum(1 for term in analytical_terms if term in query.lower())
        
        # Cap at 10
        return min(score, 10)

    def suggest_upgrade(self, current_tier: str) -> Dict[str, any]:
        """Suggest upgrade options from current tier"""
        upgrades = {
            'INSTANT': {
                'next_tier': 'STANDARD',
                'benefits': [
                    '📊 Performance trends and comparisons',
                    '📈 Historical data analysis',
                    '🎯 Key metrics and ratios'
                ],
                'estimated_time': '+8-10s'
            },
            'STANDARD': {
                'next_tier': 'DEEP',
                'benefits': [
                    '🔬 Comprehensive financial analysis',
                    '🌍 Market positioning and competition',
                    '📊 Multi-year trend analysis',
                    '💡 Investment recommendations'
                ],
                'estimated_time': '+15-25s'
            }
        }
        return upgrades.get(current_tier, {})


class ConversationContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.sessions = {}
        self.ttl = 3600  # 1 hour
    
    def update_context(
        self, 
        session_id: str, 
        query: str, 
        companies: List[str] = None,
        tickers: List[str] = None
    ):
        """Update conversation context for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'companies': set(),
                'tickers': set(),
                'queries': [],
                'created_at': datetime.now()
            }
        
        context = self.sessions[session_id]
        
        if companies:
            context['companies'].update(companies)
        if tickers:
            context['tickers'].update(tickers)
        
        context['queries'].append({
            'query': query,
            'timestamp': datetime.now()
        })
        
        context['last_query_time'] = datetime.now()
        
        # Keep only last 10 queries
        if len(context['queries']) > 10:
            context['queries'] = context['queries'][-10:]
    
    def get_context(self, session_id: str) -> Optional[Dict]:
        """Get context for session"""
        context = self.sessions.get(session_id)
        
        if not context:
            return None
        
        # Check TTL
        age = (datetime.now() - context['created_at']).total_seconds()
        if age > self.ttl:
            del self.sessions[session_id]
            return None
        
        return {
            'companies': list(context['companies']),
            'tickers': list(context['tickers']),
            'last_query_time': context.get('last_query_time'),
            'query_count': len(context['queries'])
        }
    
    def clear_context(self, session_id: str):
        """Clear context for session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️ Context cleared for session {session_id}")
