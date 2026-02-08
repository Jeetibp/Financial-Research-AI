"""
Number Extractor - Extract financial numbers from text/data
Handles various formats: "$1.5B", "₹2,50,000 Cr", "45.5%", etc.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NumberExtractor:
    """
    Extract and normalize financial numbers from text
    
    Handles:
    - Currency symbols: $, ₹, €, £
    - Scales: K, M, B, T, Cr, L
    - Formats: 1,234.56 or 1.234,56
    - Percentages: 45.5%
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scale multipliers
        self.scales = {
            'K': 1_000,
            'M': 1_000_000,
            'B': 1_000_000_000,
            'T': 1_000_000_000_000,
            'Cr': 10_000_000,  # Indian Crore
            'L': 100_000,  # Indian Lakh
            'k': 1_000,
            'm': 1_000_000,
            'b': 1_000_000_000,
            'bn': 1_000_000_000,
            'mn': 1_000_000,
            'thousand': 1_000,
            'million': 1_000_000,
            'billion': 1_000_000_000,
            'trillion': 1_000_000_000_000,
            'crore': 10_000_000,
            'lakh': 100_000
        }
    
    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract a single number from text
        
        Examples:
        - "$1.5B" → 1,500,000,000
        - "₹2,50,000 Cr" → 2,500,000,000,000
        - "45.5%" → 45.5
        - "revenue of $123.4 million" → 123,400,000
        """
        # Remove currency symbols
        text = re.sub(r'[$₹€£¥]', '', text)
        
        # Pattern: number (with commas/dots) + optional scale
        pattern = r'([\d,\.]+)\s*([KMBTkmbt]|Cr|crore|L|lakh|bn|mn|billion|million|thousand)?'
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if not matches:
            return None
        
        # Take the first match
        number_str, scale = matches[0]
        
        # Remove commas and convert to float
        try:
            number = float(number_str.replace(',', ''))
        except ValueError:
            return None
        
        # Apply scale
        if scale:
            multiplier = self.scales.get(scale, 1)
            number *= multiplier
        
        return number
    
    def extract_percentage(self, text: str) -> Optional[float]:
        """
        Extract percentage from text
        
        Examples:
        - "growth of 45.5%" → 45.5
        - "margin: 23%" → 23.0
        """
        pattern = r'([\d\.]+)\s*%'
        match = re.search(pattern, text)
        
        if match:
            return float(match.group(1))
        
        return None
    
    def extract_all_numbers(self, text: str) -> List[Tuple[float, str]]:
        """
        Extract all numbers from text with their context
        
        Returns: List of (number, context) tuples
        """
        results = []
        
        # Pattern to find numbers with context
        pattern = r'(\w+[\s\w]*?)\s*([\d,\.]+)\s*([KMBTkmbt]|Cr|L|bn|mn|%)?'
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for context, number_str, scale in matches:
            try:
                number = float(number_str.replace(',', ''))
                
                # Apply scale
                if scale and scale != '%':
                    multiplier = self.scales.get(scale, 1)
                    number *= multiplier
                
                results.append((number, context.strip()))
            except ValueError:
                continue
        
        return results
    
    def extract_financial_metrics(self, data: Dict) -> Dict[str, float]:
        """
        Extract common financial metrics from data
        
        Looks for:
        - Revenue, Sales, Turnover
        - Profit, Net Income, EBITDA
        - Assets, Liabilities, Equity
        - Price, EPS, P/E
        """
        metrics = {}
        
        # Convert data to string if it's a dict
        if isinstance(data, dict):
            text = str(data)
        else:
            text = str(data)
        
        text_lower = text.lower()
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue[:\s]+([\d,\.]+)\s*([KMBkmb])?',
            r'sales[:\s]+([\d,\.]+)\s*([KMBkmb])?',
            r'turnover[:\s]+([\d,\.]+)\s*([KMBkmb])?'
        ]
        
        for pattern in revenue_patterns:
            match = re.search(pattern, text_lower)
            if match:
                number = self.extract_number(match.group(0))
                if number:
                    metrics['revenue'] = number
                    break
        
        # Profit patterns
        profit_patterns = [
            r'net\s+income[:\s]+([\d,\.]+)\s*([KMBkmb])?',
            r'net\s+profit[:\s]+([\d,\.]+)\s*([KMBkmb])?',
            r'profit[:\s]+([\d,\.]+)\s*([KMBkmb])?',
            r'ebitda[:\s]+([\d,\.]+)\s*([KMBkmb])?'
        ]
        
        for pattern in profit_patterns:
            match = re.search(pattern, text_lower)
            if match:
                number = self.extract_number(match.group(0))
                if number:
                    metrics['profit'] = number
                    break
        
        # EPS pattern
        eps_pattern = r'eps[:\s]+([\d,\.]+)'
        match = re.search(eps_pattern, text_lower)
        if match:
            metrics['eps'] = float(match.group(1))
        
        # Stock price pattern
        price_patterns = [
            r'stock\s+price[:\s]+([\d,\.]+)',
            r'share\s+price[:\s]+([\d,\.]+)',
            r'price[:\s]+\$?([\d,\.]+)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                metrics['stock_price'] = float(match.group(1).replace(',', ''))
                break
        
        return metrics
