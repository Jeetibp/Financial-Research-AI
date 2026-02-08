"""
Enhanced Financial Calculator with Hybrid Approach
Supports:
- Type 1: Direct calculations (user provides values) - Pure Python math
- Type 2: Fetch + Calculate (real company data via yfinance)  
- Tier 1-3: Hardcoded (Top 10) → yfinance → LLM fallback

LLMs are NOT used for math - all calculations done in pure Python
"""
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel
import re

# Type 2 support
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not available - Type 2 calculations disabled")

logger = logging.getLogger(__name__)


class FinancialData(BaseModel):
    """Structured financial data"""
    metric_name: str
    value: float
    unit: str  # "USD", "INR", "percentage", "ratio"
    period: Optional[str] = None  # "Q1 2026", "FY2025"
    currency: Optional[str] = "USD"


class CalculationResult(BaseModel):
    """Result of a calculation"""
    calculation_type: str
    inputs: Dict[str, Any]
    result: Union[float, Dict[str, float]]
    unit: str
    formula: str
    interpretation: Optional[str] = None


class FinancialCalculator:
    """
    Hybrid Financial Calculator
    
    Type 1: Direct calculations (Pure Python math)
    Type 2: Company data fetching (yfinance)
    Type 3: LLM fallback for uncommon calculations
    """
    
    # Company ticker mapping for Type 2 calculations
    TICKER_MAP = {
        # Indian IT Companies
        'tcs': 'TCS.NS', 'tata consultancy': 'TCS.NS', 'infosys': 'INFY.NS',
        'wipro': 'WIPRO.NS', 'hcl tech': 'HCLTECH.NS', 'tech mahindra': 'TECHM.NS',
        # Banks
        'hdfc bank': 'HDFCBANK.NS', 'icici bank': 'ICICIBANK.NS', 'sbi': 'SBIN.NS',
        'axis bank': 'AXISBANK.NS', 'kotak mahindra': 'KOTAKBANK.NS',
        # Others
        'reliance': 'RELIANCE.NS', 'tata motors': 'TATAMOTORS.NS',
        'bharti airtel': 'BHARTIARTL.NS', 'airtel': 'BHARTIARTL.NS',
        'itc': 'ITC.NS', 'larsen': 'LT.NS', 'l&t': 'LT.NS',
        # US Stocks
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'amazon': 'AMZN', 'meta': 'META', 'tesla': 'TSLA', 'nvidia': 'NVDA'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Cache for company data
    
    # ==========================================
    # Growth & Trend Calculations
    # ==========================================
    
    def calculate_growth_rate(
        self, 
        current_value: float, 
        previous_value: float,
        periods: int = 1
    ) -> CalculationResult:
        """
        Calculate growth rate (YoY, QoQ, etc.)
        
        Formula: ((Current - Previous) / Previous) * 100
        """
        if previous_value == 0:
            return CalculationResult(
                calculation_type="growth_rate",
                inputs={"current": current_value, "previous": previous_value},
                result=0.0,
                unit="percentage",
                formula="((Current - Previous) / Previous) * 100",
                interpretation="Cannot calculate - previous value is zero"
            )
        
        growth = ((current_value - previous_value) / previous_value) * 100
        
        interpretation = self._interpret_growth(growth)
        
        return CalculationResult(
            calculation_type="growth_rate",
            inputs={
                "current_value": current_value,
                "previous_value": previous_value,
                "periods": periods
            },
            result=round(growth, 2),
            unit="percentage",
            formula="((Current - Previous) / Previous) * 100",
            interpretation=interpretation
        )
    
    def calculate_cagr(
        self, 
        starting_value: float, 
        ending_value: float, 
        num_years: float
    ) -> CalculationResult:
        """
        Calculate Compound Annual Growth Rate
        
        Formula: ((Ending / Starting) ^ (1 / Years)) - 1
        """
        if starting_value == 0 or num_years == 0:
            return CalculationResult(
                calculation_type="cagr",
                inputs={"starting": starting_value, "ending": ending_value, "years": num_years},
                result=0.0,
                unit="percentage",
                formula="((Ending / Starting) ^ (1 / Years)) - 1",
                interpretation="Cannot calculate CAGR"
            )
        
        cagr = (pow(ending_value / starting_value, 1 / num_years) - 1) * 100
        
        return CalculationResult(
            calculation_type="cagr",
            inputs={
                "starting_value": starting_value,
                "ending_value": ending_value,
                "num_years": num_years
            },
            result=round(cagr, 2),
            unit="percentage",
            formula="((Ending / Starting) ^ (1 / Years)) - 1",
            interpretation=f"Compound annual growth of {cagr:.2f}% over {num_years} years"
        )
    
    # ==========================================
    # Profitability Metrics
    # ==========================================
    
    def calculate_margin(
        self, 
        profit: float, 
        revenue: float,
        margin_type: str = "net"
    ) -> CalculationResult:
        """
        Calculate profit margins (Gross, Operating, Net, EBITDA)
        
        Formula: (Profit / Revenue) * 100
        """
        if revenue == 0:
            return CalculationResult(
                calculation_type=f"{margin_type}_margin",
                inputs={"profit": profit, "revenue": revenue},
                result=0.0,
                unit="percentage",
                formula="(Profit / Revenue) * 100",
                interpretation="Cannot calculate - revenue is zero"
            )
        
        margin = (profit / revenue) * 100
        
        return CalculationResult(
            calculation_type=f"{margin_type}_margin",
            inputs={"profit": profit, "revenue": revenue},
            result=round(margin, 2),
            unit="percentage",
            formula="(Profit / Revenue) * 100",
            interpretation=self._interpret_margin(margin, margin_type)
        )
    
    def calculate_roa(self, net_income: float, total_assets: float) -> CalculationResult:
        """
        Return on Assets
        
        Formula: (Net Income / Total Assets) * 100
        """
        if total_assets == 0:
            return self._zero_result("roa", net_income, total_assets)
        
        roa = (net_income / total_assets) * 100
        
        return CalculationResult(
            calculation_type="roa",
            inputs={"net_income": net_income, "total_assets": total_assets},
            result=round(roa, 2),
            unit="percentage",
            formula="(Net Income / Total Assets) * 100"
        )
    
    def calculate_roe(self, net_income: float, shareholders_equity: float) -> CalculationResult:
        """
        Return on Equity
        
        Formula: (Net Income / Shareholders Equity) * 100
        """
        if shareholders_equity == 0:
            return self._zero_result("roe", net_income, shareholders_equity)
        
        roe = (net_income / shareholders_equity) * 100
        
        return CalculationResult(
            calculation_type="roe",
            inputs={"net_income": net_income, "shareholders_equity": shareholders_equity},
            result=round(roe, 2),
            unit="percentage",
            formula="(Net Income / Shareholders Equity) * 100",
            interpretation=self._interpret_roe(roe)
        )
    
    # ==========================================
    # Valuation Ratios
    # ==========================================
    
    def calculate_pe_ratio(self, stock_price: float, eps: float) -> CalculationResult:
        """
        Price-to-Earnings Ratio
        
        Formula: Stock Price / EPS
        """
        if eps == 0:
            return self._zero_result("pe_ratio", stock_price, eps)
        
        pe = stock_price / eps
        
        return CalculationResult(
            calculation_type="pe_ratio",
            inputs={"stock_price": stock_price, "eps": eps},
            result=round(pe, 2),
            unit="ratio",
            formula="Stock Price / EPS",
            interpretation=self._interpret_pe(pe)
        )
    
    def calculate_pb_ratio(self, stock_price: float, book_value_per_share: float) -> CalculationResult:
        """
        Price-to-Book Ratio
        
        Formula: Stock Price / Book Value per Share
        """
        if book_value_per_share == 0:
            return self._zero_result("pb_ratio", stock_price, book_value_per_share)
        
        pb = stock_price / book_value_per_share
        
        return CalculationResult(
            calculation_type="pb_ratio",
            inputs={"stock_price": stock_price, "book_value_per_share": book_value_per_share},
            result=round(pb, 2),
            unit="ratio",
            formula="Stock Price / Book Value per Share"
        )
    
    def calculate_market_cap(self, stock_price: float, shares_outstanding: float) -> CalculationResult:
        """
        Market Capitalization
        
        Formula: Stock Price * Shares Outstanding
        """
        market_cap = stock_price * shares_outstanding
        
        return CalculationResult(
            calculation_type="market_cap",
            inputs={"stock_price": stock_price, "shares_outstanding": shares_outstanding},
            result=market_cap,
            unit="currency",
            formula="Stock Price * Shares Outstanding",
            interpretation=self._interpret_market_cap(market_cap)
        )
    
    # ==========================================
    # Liquidity Ratios
    # ==========================================
    
    def calculate_current_ratio(self, current_assets: float, current_liabilities: float) -> CalculationResult:
        """
        Current Ratio (Liquidity measure)
        
        Formula: Current Assets / Current Liabilities
        """
        if current_liabilities == 0:
            return self._zero_result("current_ratio", current_assets, current_liabilities)
        
        ratio = current_assets / current_liabilities
        
        return CalculationResult(
            calculation_type="current_ratio",
            inputs={"current_assets": current_assets, "current_liabilities": current_liabilities},
            result=round(ratio, 2),
            unit="ratio",
            formula="Current Assets / Current Liabilities",
            interpretation=self._interpret_current_ratio(ratio)
        )
    
    def calculate_quick_ratio(
        self, 
        current_assets: float, 
        inventory: float, 
        current_liabilities: float
    ) -> CalculationResult:
        """
        Quick Ratio (Acid Test)
        
        Formula: (Current Assets - Inventory) / Current Liabilities
        """
        if current_liabilities == 0:
            return self._zero_result("quick_ratio", current_assets - inventory, current_liabilities)
        
        ratio = (current_assets - inventory) / current_liabilities
        
        return CalculationResult(
            calculation_type="quick_ratio",
            inputs={
                "current_assets": current_assets,
                "inventory": inventory,
                "current_liabilities": current_liabilities
            },
            result=round(ratio, 2),
            unit="ratio",
            formula="(Current Assets - Inventory) / Current Liabilities"
        )
    
    # ==========================================
    # Efficiency Ratios
    # ==========================================
    
    def calculate_asset_turnover(self, revenue: float, total_assets: float) -> CalculationResult:
        """
        Asset Turnover Ratio
        
        Formula: Revenue / Total Assets
        """
        if total_assets == 0:
            return self._zero_result("asset_turnover", revenue, total_assets)
        
        ratio = revenue / total_assets
        
        return CalculationResult(
            calculation_type="asset_turnover",
            inputs={"revenue": revenue, "total_assets": total_assets},
            result=round(ratio, 2),
            unit="ratio",
            formula="Revenue / Total Assets"
        )
    
    def calculate_inventory_turnover(self, cogs: float, avg_inventory: float) -> CalculationResult:
        """
        Inventory Turnover
        
        Formula: COGS / Average Inventory
        """
        if avg_inventory == 0:
            return self._zero_result("inventory_turnover", cogs, avg_inventory)
        
        ratio = cogs / avg_inventory
        
        return CalculationResult(
            calculation_type="inventory_turnover",
            inputs={"cogs": cogs, "avg_inventory": avg_inventory},
            result=round(ratio, 2),
            unit="ratio",
            formula="COGS / Average Inventory"
        )
    
    # ==========================================
    # Leverage Ratios
    # ==========================================
    
    def calculate_debt_to_equity(self, total_debt: float, shareholders_equity: float) -> CalculationResult:
        """
        Debt-to-Equity Ratio
        
        Formula: Total Debt / Shareholders Equity
        """
        if shareholders_equity == 0:
            return self._zero_result("debt_to_equity", total_debt, shareholders_equity)
        
        ratio = total_debt / shareholders_equity
        
        return CalculationResult(
            calculation_type="debt_to_equity",
            inputs={"total_debt": total_debt, "shareholders_equity": shareholders_equity},
            result=round(ratio, 2),
            unit="ratio",
            formula="Total Debt / Shareholders Equity",
            interpretation=self._interpret_debt_equity(ratio)
        )
    
    def calculate_debt_ratio(self, total_debt: float, total_assets: float) -> CalculationResult:
        """
        Debt Ratio
        
        Formula: Total Debt / Total Assets
        """
        if total_assets == 0:
            return self._zero_result("debt_ratio", total_debt, total_assets)
        
        ratio = total_debt / total_assets
        
        return CalculationResult(
            calculation_type="debt_ratio",
            inputs={"total_debt": total_debt, "total_assets": total_assets},
            result=round(ratio, 2),
            unit="ratio",
            formula="Total Debt / Total Assets"
        )
    
    # ==========================================
    # TYPE 2: COMPANY DATA FETCHING (yfinance)
    # ==========================================
    
    def get_company_ticker(self, company_name: str) -> Optional[str]:
        """Convert company name to ticker symbol"""
        if not YFINANCE_AVAILABLE:
            self.logger.warning("yfinance not available for ticker resolution")
            return None
        
        company_lower = company_name.lower().strip()
        
        # Exact match
        if company_lower in self.TICKER_MAP:
            return self.TICKER_MAP[company_lower]
        
        # Partial match
        for key, ticker in self.TICKER_MAP.items():
            if key in company_lower or company_lower in key:
                return ticker
        
        # Default for Indian stocks
        if '.' not in company_name:
            return f"{company_name.upper()}.NS"
        
        return company_name.upper()
    
    def fetch_company_metrics(self, company_name: str) -> Optional[Dict]:
        """
        Fetch comprehensive financial metrics from yfinance
        Type 2: Real company data
        """
        if not YFINANCE_AVAILABLE:
            self.logger.error("yfinance is not installed. Install with: pip install yfinance")
            return None
        
        try:
            ticker = self.get_company_ticker(company_name)
            if not ticker:
                return None
            
            # Check cache
            cache_key = f"{ticker}_metrics"
            if cache_key in self.cache:
                self.logger.info(f"Using cached data for {ticker}")
                return self.cache[cache_key]
            
            self.logger.info(f"🔍 Fetching data for {company_name} ({ticker})")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {
                'company': company_name,
                'ticker': ticker,
                'current_price': info.get('currentPrice'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                
                # Valuation
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                
                # Profitability
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                
                # Financial Health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                
                # Per Share
                'eps': info.get('trailingEps'),
                'book_value': info.get('bookValue'),
                
                # Additional
                'beta': info.get('beta'),
                'dividend_yield': info.get('dividendYield')
            }
            
            # Cache for future use
            self.cache[cache_key] = metrics
            self.logger.info(f"✅ Successfully fetched data for {company_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {company_name}: {e}")
            return None
    
    def calculate_metric_for_company(
        self, 
        company_name: str, 
        metric: str
    ) -> Optional[CalculationResult]:
        """
        Calculate specific metric for a company (Type 2)
        
        Args:
            company_name: Company name (e.g., 'TCS', 'Infosys', 'Apple')
            metric: Metric to calculate ('roe', 'pe_ratio', 'debt_to_equity', etc.)
        
        Returns:
            CalculationResult with fetched data or None
        """
        metrics = self.fetch_company_metrics(company_name)
        if not metrics:
            return None
        
        metric_lower = metric.lower()
        
        # P/E Ratio
        if metric_lower in ['pe', 'p/e', 'pe_ratio', 'pe ratio']:
            pe = metrics.get('pe_ratio')
            if pe:
                result = CalculationResult(
                    calculation_type="pe_ratio_company",
                    inputs={'company': company_name, 'ticker': metrics['ticker']},
                    result=round(pe, 2),
                    unit="ratio",
                    formula="Stock Price / EPS",
                    interpretation=f"{self._interpret_pe(pe)} | Price: ₹{metrics.get('current_price', 0):.2f} | EPS: ₹{metrics.get('eps', 0):.2f}"
                )
                result.source = 'Yahoo Finance'
                result.sector = metrics.get('sector')
                return result
        
        # ROE
        elif metric_lower in ['roe', 'return on equity']:
            roe_decimal = metrics.get('roe')
            if roe_decimal:
                roe = roe_decimal * 100  # Convert to percentage
                result = CalculationResult(
                    calculation_type="roe_company",
                    inputs={'company': company_name, 'ticker': metrics['ticker']},
                    result=round(roe, 2),
                    unit="percentage",
                    formula="(Net Income / Shareholder Equity) × 100",
                    interpretation=self._interpret_roe(roe)
                )
                result.source = 'Yahoo Finance'
                result.sector = metrics.get('sector')
                return result
        
        # Debt-to-Equity
        elif metric_lower in ['debt_to_equity', 'd/e', 'debt to equity']:
            d_e = metrics.get('debt_to_equity')
            if d_e:
                # Yahoo returns as percentage, convert to ratio
                d_e_ratio = d_e / 100 if d_e > 10 else d_e
                result = CalculationResult(
                    calculation_type="debt_to_equity_company",
                    inputs={'company': company_name, 'ticker': metrics['ticker']},
                    result=round(d_e_ratio, 2),
                    unit="ratio",
                    formula="Total Debt / Shareholder Equity",
                    interpretation=self._interpret_debt_equity(d_e_ratio)
                )
                result.source = 'Yahoo Finance'
                return result
        
        # Profit Margin
        elif metric_lower in ['profit_margin', 'net_margin', 'profit margin']:
            margin = metrics.get('profit_margin')
            if margin:
                margin_pct = margin * 100
                result = CalculationResult(
                    calculation_type="profit_margin_company",
                    inputs={'company': company_name, 'ticker': metrics['ticker']},
                    result=round(margin_pct, 2),
                    unit="percentage",
                    formula="(Net Profit / Revenue) × 100",
                    interpretation=self._interpret_margin(margin_pct, 'net')
                )
                result.source = 'Yahoo Finance'
                return result
        
        # Current Ratio
        elif metric_lower in ['current_ratio', 'current ratio']:
            ratio = metrics.get('current_ratio')
            if ratio:
                result = CalculationResult(
                    calculation_type="current_ratio_company",
                    inputs={'company': company_name, 'ticker': metrics['ticker']},
                    result=round(ratio, 2),
                    unit="ratio",
                    formula="Current Assets / Current Liabilities",
                    interpretation=self._interpret_current_ratio(ratio)
                )
                result.source = 'Yahoo Finance'
                return result
        
        return None
    
    # ==========================================
    # Helper Methods
    # ==========================================
    
    def _zero_result(self, calc_type: str, *inputs) -> CalculationResult:
        """Handle division by zero cases"""
        return CalculationResult(
            calculation_type=calc_type,
            inputs={"values": inputs},
            result=0.0,
            unit="ratio",
            formula="N/A - Division by zero",
            interpretation="Cannot calculate - denominator is zero"
        )
    
    def _interpret_growth(self, growth: float) -> str:
        """Interpret growth rate"""
        if growth > 50:
            return f"Exceptional growth of {growth:.1f}%"
        elif growth > 20:
            return f"Strong growth of {growth:.1f}%"
        elif growth > 10:
            return f"Solid growth of {growth:.1f}%"
        elif growth > 0:
            return f"Modest growth of {growth:.1f}%"
        elif growth > -5:
            return f"Slight decline of {abs(growth):.1f}%"
        else:
            return f"Significant decline of {abs(growth):.1f}%"
    
    def _interpret_margin(self, margin: float, margin_type: str) -> str:
        """Interpret profit margin"""
        if margin < 0:
            return f"Negative {margin_type} margin - operating at a loss"
        elif margin < 5:
            return f"Low {margin_type} margin of {margin:.1f}%"
        elif margin < 15:
            return f"Moderate {margin_type} margin of {margin:.1f}%"
        elif margin < 25:
            return f"Healthy {margin_type} margin of {margin:.1f}%"
        else:
            return f"Excellent {margin_type} margin of {margin:.1f}%"
    
    def _interpret_roe(self, roe: float) -> str:
        """Interpret Return on Equity"""
        if roe > 20:
            return f"Excellent ROE of {roe:.1f}% - highly profitable"
        elif roe > 15:
            return f"Good ROE of {roe:.1f}%"
        elif roe > 10:
            return f"Average ROE of {roe:.1f}%"
        else:
            return f"Below average ROE of {roe:.1f}%"
    
    def _interpret_pe(self, pe: float) -> str:
        """Interpret P/E ratio"""
        if pe < 0:
            return "Negative P/E - company is unprofitable"
        elif pe < 15:
            return f"P/E of {pe:.1f} - potentially undervalued"
        elif pe < 25:
            return f"P/E of {pe:.1f} - fairly valued"
        elif pe < 40:
            return f"P/E of {pe:.1f} - premium valuation"
        else:
            return f"P/E of {pe:.1f} - highly expensive"
    
    def _interpret_current_ratio(self, ratio: float) -> str:
        """Interpret current ratio"""
        if ratio < 1:
            return f"Current ratio {ratio:.2f} - liquidity concerns"
        elif ratio < 1.5:
            return f"Current ratio {ratio:.2f} - adequate liquidity"
        elif ratio < 3:
            return f"Current ratio {ratio:.2f} - good liquidity"
        else:
            return f"Current ratio {ratio:.2f} - very strong liquidity"
    
    def _interpret_debt_equity(self, ratio: float) -> str:
        """Interpret debt-to-equity ratio"""
        if ratio < 0.5:
            return f"Low leverage ({ratio:.2f}) - conservative"
        elif ratio < 1:
            return f"Moderate leverage ({ratio:.2f})"
        elif ratio < 2:
            return f"High leverage ({ratio:.2f})"
        else:
            return f"Very high leverage ({ratio:.2f}) - risky"
    
    def _interpret_market_cap(self, market_cap: float) -> str:
        """Interpret market capitalization"""
        if market_cap > 200_000_000_000:  # >$200B
            return "Mega-cap company"
        elif market_cap > 10_000_000_000:  # >$10B
            return "Large-cap company"
        elif market_cap > 2_000_000_000:  # >$2B
            return "Mid-cap company"
        elif market_cap > 300_000_000:  # >$300M
            return "Small-cap company"
        else:
            return "Micro-cap company"
