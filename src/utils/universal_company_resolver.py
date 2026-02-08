"""
Universal Company Resolver - Works for ANY company worldwide
Uses multiple data sources: OpenAI + yfinance + web search
"""

import os
import json
import yfinance as yf
import requests
from typing import Dict, List, Optional
from openai import OpenAI
from src.utils.logger import get_logger

logger = get_logger("universal_resolver")


class UniversalCompanyResolver:
    """
    Generic company resolution for ANY stock worldwide
    No manual mappings required
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    def resolve_company(self, query: str, context: str = "") -> Dict:
        """
        Universal company resolution using AI + real-time data
        
        Strategy:
        1. Use OpenAI to extract company name and guess ticker
        2. Validate ticker with yfinance (real-time market data)
        3. If validation fails, search web for correct ticker
        4. Return validated company info
        """
        
        logger.info(f"Resolving company from query")
        
        # Step 1: AI extraction of company name and ticker guess
        ai_extraction = self._ai_extract_company(query, context)
        
        if not ai_extraction.get("companies"):
            return {
                "resolved": False,
                "message": "No company detected in query"
            }
        
        # Step 2: Validate and enrich each company
        validated_companies = []
        for company in ai_extraction["companies"]:
            validated = self._validate_and_enrich_company(company)
            if validated.get("valid"):
                validated_companies.append(validated)
        
        if not validated_companies:
            return {
                "resolved": False,
                "message": "Could not find valid ticker for mentioned companies",
                "extracted": ai_extraction["companies"]
            }
        
        # Step 3: Build response
        return self._build_resolution_response(validated_companies)
    
    def _ai_extract_company(self, query: str, context: str = "") -> Dict:
        """
        Use OpenAI to extract company names from query
        Works for ANY company worldwide
        """
        
        system_prompt = """You are a global financial market expert.

Extract ALL company names mentioned in the user's query.

RULES:
1. Extract the ACTUAL company name (not brand/subsidiary if parent is traded)
2. Determine the most likely stock ticker symbol
3. Identify the country/exchange
4. Handle subsidiaries by identifying the publicly traded parent

Examples:

Query: "jio stock analysis"
Response:
{
  "companies": [
    {
      "mentioned_name": "jio",
      "actual_company": "Reliance Industries Limited",
      "ticker_guess": "RELIANCE.NS",
      "country": "India",
      "exchange": "NSE",
      "reason": "Jio is a subsidiary of Reliance Industries which is publicly traded on NSE"
    }
  ]
}

Query: "compare apple and samsung stocks"
Response:
{
  "companies": [
    {
      "mentioned_name": "apple",
      "actual_company": "Apple Inc.",
      "ticker_guess": "AAPL",
      "country": "USA",
      "exchange": "NASDAQ",
      "reason": "Apple Inc. trades on NASDAQ"
    },
    {
      "mentioned_name": "samsung",
      "actual_company": "Samsung Electronics Co., Ltd.",
      "ticker_guess": "005930.KS",
      "country": "South Korea",
      "exchange": "KRX",
      "reason": "Samsung Electronics trades on Korea Exchange"
    }
  ]
}

Query: "tata motors stock price"
Response:
{
  "companies": [
    {
      "mentioned_name": "tata motors",
      "actual_company": "Tata Motors Limited",
      "ticker_guess": "TATAMOTORS.NS",
      "country": "India",
      "exchange": "NSE",
      "reason": "Tata Motors trades on NSE and BSE"
    }
  ]
}

CRITICAL: Return valid JSON with "companies" array. If no company found, return empty array."""

        user_message = f"""Query: "{query}"
Context from conversation: {context if context else "None"}

Extract all companies mentioned."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"AI extracted {len(result.get('companies', []))} companies")
            return result
            
        except Exception as e:
            logger.error(f"AI extraction error: {str(e)}")
            return {"companies": []}
    
    def _validate_and_enrich_company(self, company: Dict) -> Dict:
        """
        Validate ticker and enrich with real-time market data
        """
        
        ticker_guess = company.get("ticker_guess", "")
        company_name = company.get("actual_company", "")
        
        logger.info(f"Validating ticker: {ticker_guess}")
        
        # If no ticker provided, skip validation
        if not ticker_guess:
            logger.warning(f"No ticker found for {company_name}, skipping validation")
            return {"valid": False, **company, "reason": "No ticker available"}
        
        # Try primary ticker
        validation = self._yfinance_validate(ticker_guess)
        if validation.get("valid"):
            return {**company, **validation}
        
        # Try alternative formats
        alternatives = self._generate_ticker_alternatives(ticker_guess, company)
        for alt_ticker in alternatives:
            validation = self._yfinance_validate(alt_ticker)
            if validation.get("valid"):
                logger.info(f"Found working alternative: {alt_ticker}")
                return {**company, "ticker_guess": alt_ticker, **validation}
        
        # Last resort: Web search for ticker
        web_ticker = self._web_search_ticker(company_name, company.get("country"))
        if web_ticker:
            validation = self._yfinance_validate(web_ticker)
            if validation.get("valid"):
                logger.info(f"Web search found: {web_ticker}")
                return {**company, "ticker_guess": web_ticker, **validation}
        
        # Failed all attempts
        logger.warning(f"Could not validate {company_name}")
        return {"valid": False, **company}
    
    def _yfinance_validate(self, ticker: str) -> Dict:
        """Validate ticker using yfinance"""
        if not ticker:
            return {"valid": False}
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if data exists
            if not info or len(info) < 5:
                return {"valid": False}
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            return {
                "valid": True,
                "ticker": ticker,
                "company_name": info.get('longName', info.get('shortName', '')),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0),
                "currency": info.get('currency', 'USD'),
                "current_price": current_price,
                "exchange": info.get('exchange', ''),
                "country": info.get('country', ''),
                "website": info.get('website', ''),
                "description": info.get('longBusinessSummary', '')[:500]
            }
            
        except Exception as e:
            logger.debug(f"yfinance validation failed for {ticker}: {str(e)}")
            return {"valid": False}
    
    def _generate_ticker_alternatives(self, ticker: str, company: Dict) -> List[str]:
        """
        Generate alternative ticker formats for different exchanges
        """
        
        # Handle None or empty ticker
        if not ticker:
            return []
        
        alternatives = []
        base_ticker = ticker.split('.')[0]  # Remove suffix
        country = company.get("country", "").lower()
        
        # Indian stocks
        if "india" in country:
            alternatives.extend([
                f"{base_ticker}.NS",  # NSE
                f"{base_ticker}.BO",  # BSE
            ])
        
        # US stocks
        elif "usa" in country or "us" in country or "america" in country:
            alternatives.append(base_ticker)  # No suffix for US
        
        # UK stocks
        elif "uk" in country or "britain" in country:
            alternatives.append(f"{base_ticker}.L")
        
        # Japanese stocks
        elif "japan" in country:
            alternatives.append(f"{base_ticker}.T")
        
        # Hong Kong stocks
        elif "hong kong" in country or "hk" in country:
            alternatives.append(f"{base_ticker}.HK")
        
        # Korean stocks
        elif "korea" in country:
            alternatives.append(f"{base_ticker}.KS")
        
        # Chinese stocks
        elif "china" in country:
            alternatives.extend([
                f"{base_ticker}.SS",  # Shanghai
                f"{base_ticker}.SZ",  # Shenzhen
            ])
        
        # Australian stocks
        elif "australia" in country:
            alternatives.append(f"{base_ticker}.AX")
        
        # German stocks
        elif "germany" in country:
            alternatives.extend([
                f"{base_ticker}.DE",  # XETRA
                f"{base_ticker}.F",   # Frankfurt
            ])
        
        # Canadian stocks
        elif "canada" in country:
            alternatives.append(f"{base_ticker}.TO")
        
        # Try common formats as fallback
        alternatives.extend([
            ticker,
            base_ticker,
            f"{base_ticker}.NS",
            f"{base_ticker}.BO"
        ])
        
        # Remove duplicates
        return list(dict.fromkeys(alternatives))
    
    def _web_search_ticker(self, company_name: str, country: str = "") -> Optional[str]:
        """
        Search web for correct ticker symbol using Tavily
        """
        if not self.tavily_api_key:
            logger.debug("No Tavily API key, skipping web search")
            return None
        
        search_query = f"{company_name} stock ticker symbol {country}"
        
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key,
                    "query": search_query,
                    "max_results": 3
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                # Ask OpenAI to extract ticker from search results
                ticker = self._extract_ticker_from_search(results, company_name)
                return ticker
                
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
        
        return None
    
    def _extract_ticker_from_search(self, search_results: List[Dict], company_name: str) -> Optional[str]:
        """Use AI to extract ticker from search results"""
        
        search_text = "\n\n".join([
            f"Title: {r.get('title', '')}\nContent: {r.get('content', '')}"
            for r in search_results[:3]
        ])
        
        prompt = f"""Extract the stock ticker symbol for {company_name} from these search results:

{search_text}

Return ONLY the ticker symbol in format: SYMBOL or SYMBOL.EXCHANGE
Examples: AAPL, RELIANCE.NS, 005930.KS

Ticker:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            
            ticker = response.choices[0].message.content.strip()
            logger.info(f"Extracted ticker from web: {ticker}")
            return ticker
            
        except Exception as e:
            logger.error(f"Ticker extraction error: {str(e)}")
            return None
    
    def _build_resolution_response(self, validated_companies: List[Dict]) -> Dict:
        """Build final resolution response"""
        
        if len(validated_companies) == 1:
            company = validated_companies[0]
            return {
                "resolved": True,
                "company": company.get("company_name"),
                "ticker": company.get("ticker"),
                "sector": company.get("sector"),
                "industry": company.get("industry"),
                "country": company.get("country"),
                "exchange": company.get("exchange"),
                "market_cap": company.get("market_cap"),
                "current_price": company.get("current_price"),
                "currency": company.get("currency"),
                "description": company.get("description"),
                "is_subsidiary": company.get("mentioned_name") != company.get("actual_company").lower(),
                "original_mention": company.get("mentioned_name"),
                "note": company.get("reason", "")
            }
        else:
            # Multiple companies
            return {
                "resolved": True,
                "multiple": True,
                "companies": [
                    {
                        "company": c.get("company_name"),
                        "ticker": c.get("ticker"),
                        "sector": c.get("sector"),
                        "country": c.get("country"),
                        "current_price": c.get("current_price"),
                        "currency": c.get("currency")
                    }
                    for c in validated_companies
                ],
                "note": f"Found {len(validated_companies)} companies for analysis"
            }
