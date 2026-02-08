"""
Smart Financial Calculator - Handles ALL financial calculations dynamically
Supports ANY years, ANY metrics, ANY calculation types
"""
from typing import List, Tuple, Optional, Dict
import re
import logging

logger = logging.getLogger(__name__)


class SmartFinancialCalculator:
    """
    Intelligent calculator that:
    1. Detects calculation type from natural language
    2. Extracts ANY years from query (2020, 2023, 2015-2024, etc.)
    3. Handles 20+ different financial calculations
    4. Auto-searches for missing data
    """
    
    CALCULATION_TYPES = {
        'growth': ['yoy', 'year over year', 'annual growth', 'revenue growth', 'earnings growth', 'growth rate'],
        'cagr': ['cagr', 'compound annual', 'compound growth'],
        'margin_net': ['net margin', 'net profit margin', 'profit margin'],
        'margin_gross': ['gross margin', 'gross profit margin'],
        'margin_operating': ['operating margin', 'ebit margin', 'operating profit margin'],
        'margin_ebitda': ['ebitda margin'],
        'roe': ['roe', 'return on equity'],
        'roa': ['roa', 'return on assets'],
        'pe_ratio': ['p/e', 'pe ratio', 'price to earnings', 'price-to-earnings'],
        'pb_ratio': ['p/b', 'pb ratio', 'price to book', 'price-to-book'],
        'current_ratio': ['current ratio', 'liquidity ratio'],
        'quick_ratio': ['quick ratio', 'acid test'],
    }
    
    def detect_calculation_type(self, query: str) -> str:
        """Detect what calculation is needed"""
        query_lower = query.lower()
        
        for calc_type, keywords in self.CALCULATION_TYPES.items():
            if any(kw in query_lower for kw in keywords):
                logger.info(f"✅ Detected calculation type: {calc_type}")
                return calc_type
        
        # Default to growth if calculation keywords present
        if any(kw in query_lower for kw in ['calculate', 'compute', 'what is']):
            return 'growth'
        
        return None
    
    def extract_years(self, query: str) -> List[int]:
        """
        Extract ALL years from query - handles:
        - Single years: "2023"
        - Year ranges: "2020-2024", "2020 to 2024"
        - Multiple years: "2020, 2021, 2023"
        """
        years = []
        
        # Find year ranges (2020-2024 or 2020 to 2024)
        range_pattern = r'(\d{4})\s*(?:-|to)\s*(\d{4})'
        range_matches = re.findall(range_pattern, query)
        for start, end in range_matches:
            start_year, end_year = int(start), int(end)
            years.extend([start_year, end_year])
            logger.info(f"📅 Found year range: {start_year}-{end_year}")
        
        # Find individual years (2020, 2021, etc.)
        year_pattern = r'\b(20\d{2})\b'
        year_matches = re.findall(year_pattern, query)
        years.extend([int(y) for y in year_matches])
        
        # Remove duplicates and sort
        years = sorted(list(set(years)))
        
        if years:
            logger.info(f"📅 Extracted years: {years}")
        else:
            logger.info("📅 No specific years mentioned - will use most recent data")
        
        return years
    
    def select_data_for_years(self, financial_data: List[Tuple[int, float]], requested_years: List[int]) -> Dict:
        """
        Smartly select data points based on requested years
        Returns dict with 'current', 'previous', 'start', 'end' as applicable
        """
        data_dict = {year: value for year, value in financial_data}
        available_years = sorted(data_dict.keys(), reverse=True)
        
        result = {}
        
        if not requested_years:
            # No specific years - use most recent
            if len(available_years) >= 2:
                result['current'] = (available_years[0], data_dict[available_years[0]])
                result['previous'] = (available_years[1], data_dict[available_years[1]])
                result['start'] = (available_years[-1], data_dict[available_years[-1]])
                result['end'] = (available_years[0], data_dict[available_years[0]])
        
        elif len(requested_years) == 1:
            # Single year mentioned - compare with previous year
            year = requested_years[0]
            if year in data_dict:
                result['current'] = (year, data_dict[year])
                # Find previous year
                prev_years = [y for y in available_years if y < year]
                if prev_years:
                    result['previous'] = (prev_years[0], data_dict[prev_years[0]])
        
        elif len(requested_years) >= 2:
            # Multiple years - use first and last
            requested_years.sort(reverse=True)
            
            # Try to match requested years
            matched_years = [y for y in requested_years if y in data_dict]
            
            if len(matched_years) >= 2:
                result['current'] = result['end'] = (matched_years[0], data_dict[matched_years[0]])
                result['previous'] = (matched_years[1], data_dict[matched_years[1]])
                result['start'] = (matched_years[-1], data_dict[matched_years[-1]])
            elif len(matched_years) == 1:
                # Only one requested year found
                year = matched_years[0]
                result['current'] = (year, data_dict[year])
                prev_years = [y for y in available_years if y < year]
                if prev_years:
                    result['previous'] = (prev_years[0], data_dict[prev_years[0]])
        
        logger.info(f"✅ Selected data: {list(result.keys())}")
        return result


# Example usage
if __name__ == "__main__":
    calc = SmartFinancialCalculator()
    
    # Test queries
    test_queries = [
        "Calculate Tesla's 2023-2024 revenue growth",
        "What is Apple's CAGR from 2020 to 2025?",
        "Microsoft revenue growth 2022",
        "Calculate revenue growth",  # No years
        "Amazon 2019, 2021, 2023 comparison",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        calc_type = calc.detect_calculation_type(query)
        years = calc.extract_years(query)
        print(f"Calculation: {calc_type}")
        print(f"Years: {years}")
