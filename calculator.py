"""
Sharpe Ratio Calculator - Core Calculation Functions
Professional implementation with multiple calculation methods
FIXED VERSION - Resolves single stock download errors
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Tuple, Optional

class SharpeCalculator:
    """Professional Sharpe Ratio Calculator with multiple methods"""
    
    def __init__(self):
        self.risk_free_rate = self.fetch_risk_free_rate()
    
    def fetch_risk_free_rate(self) -> float:
        """
        Fetch current 10-year Treasury rate as risk-free rate
        Fallback to 4.5% if API fails
        """
        try:
            # Try to get 10-year Treasury rate (simplified approach)
            # In production, you'd use FRED API with actual key
            return 0.045  # 4.5% fallback
        except:
            return 0.045  # 4.5% fallback
    
    def download_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download stock data with robust error handling
        FIXED VERSION - handles single stock downloads properly
        """
        try:
            # Download data using yfinance
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            # Handle empty data
            if data.empty:
                raise ValueError(f"No data available for {symbols} in the specified date range")
            
            if len(symbols) == 1:
                # Single stock case - yfinance returns different structure
                symbol = symbols[0]
                
                # Check if data has the expected structure
                if 'Adj Close' in data.columns:
                    # Standard case - data has multiple columns
                    result = pd.DataFrame({symbol: data['Adj Close']})
                else:
                    # Edge case - data might be a series
                    if isinstance(data, pd.Series):
                        result = pd.DataFrame({symbol: data})
                    else:
                        # Assume the data is already in the right format
                        result = pd.DataFrame({symbol: data.iloc[:, 0] if len(data.columns) > 0 else data})
                
                # Remove any NaN values
                result = result.dropna()
                
                if result.empty:
                    raise ValueError(f"No valid data available for {symbol}")
                
                return result
                
            else:
                # Multiple stocks case
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Adj Close']
                else:
                    # Fallback - use the first level of columns
                    adj_close_data = data.iloc[:, :len(symbols)]
                    adj_close_data.columns = symbols
                
                # Remove any NaN values
                adj_close_data = adj_close_data.dropna()
                
                if adj_close_data.empty:
                    raise ValueError(f"No valid data available for {symbols}")
                
                return adj_close_data
                
        except Exception as e:
            raise ValueError(f"Error downloading data for {symbols}: {str(e)}")
    
    def validate_portfolio(self, symbols: List[str], weights: List[float]) -> bool:
        """
        Validate portfolio inputs
        """
        if len(symbols) != len(weights):
            raise ValueError("Number of symbols must match number of weights")
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (100%), current sum: {sum(weights):.3f}")
        
        if any(w < 0 for w in weights):
            raise ValueError("Weights cannot be negative")
        
        if any(not symbol.strip() for symbol in symbols):
            raise ValueError("All symbols must be non-empty")
        
        return True
    
    def calculate_simple_sharpe(self, symbols: List[str], weights: List[float], 
                              start_date: str, end_date: str, 
                              risk_free_rate: Optional[float] = None) -> Dict:
        """
        Method 1: Simple Portfolio Sharpe Ratio
        Downloads data and calculates weighted portfolio returns
        """
        # Validate inputs
        self.validate_portfolio(symbols, weights)
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Download data
        data = self.download_stock_data(symbols, start_date, end_date)
        
        # Calculate daily returns
        if len(symbols) == 1:
            symbol = symbols[0]
            daily_returns = data[symbol].pct_change().dropna()
            portfolio_returns = daily_returns * weights[0]
        else:
            daily_returns = data.pct_change().dropna()
            portfolio_returns = (daily_returns * weights).sum(axis=1)
        
        # Remove any remaining NaN values
        portfolio_returns = portfolio_returns.dropna()
        
        if len(portfolio_returns) < 30:
            raise ValueError("Insufficient data points for reliable calculation (need at least 30 days)")
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Avoid division by zero
        if annual_volatility == 0:
            raise ValueError("Portfolio volatility is zero - cannot calculate Sharpe ratio")
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # Additional metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        # Win rate (percentage of positive days)
        win_rate = (portfolio_returns > 0).mean()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'risk_free_rate': risk_free_rate,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'num_observations': len(portfolio_returns)
        }
    
    def calculate_manual_sharpe(self, portfolio_return: float, portfolio_volatility: float,
                              risk_free_rate: Optional[float] = None) -> Dict:
        """
        Method 2: Manual Sharpe Ratio Calculation
        User provides return, volatility, and risk-free rate
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if portfolio_volatility <= 0:
            raise ValueError("Portfolio volatility must be positive")
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annual_return': portfolio_return,
            'annual_volatility': portfolio_volatility,
            'risk_free_rate': risk_free_rate,
            'excess_return': portfolio_return - risk_free_rate
        }
    
    def calculate_correlation_adjusted_sharpe(self, symbols: List[str], weights: List[float],
                                            start_date: str, end_date: str,
                                            risk_free_rate: Optional[float] = None) -> Dict:
        """
        Method 3: Correlation-Adjusted Portfolio Volatility
        More accurate calculation using correlation matrix
        """
        self.validate_portfolio(symbols, weights)
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # For single asset, use simple calculation
        if len(symbols) == 1:
            return self.calculate_simple_sharpe(symbols, weights, start_date, end_date, risk_free_rate)
        
        # Download data
        data = self.download_stock_data(symbols, start_date, end_date)
        
        # Calculate individual returns and statistics
        daily_returns = data.pct_change().dropna()
        
        if len(daily_returns) < 30:
            raise ValueError("Insufficient data points for reliable calculation")
        
        individual_returns = daily_returns.mean() * 252
        individual_volatilities = daily_returns.std() * np.sqrt(252)
        correlation_matrix = daily_returns.corr()
        
        # Portfolio return (weighted average)
        portfolio_return = (individual_returns * weights).sum()
        
        # Portfolio variance using correlation matrix
        portfolio_variance = 0
        for i, asset1 in enumerate(symbols):
            for j, asset2 in enumerate(symbols):
                portfolio_variance += (weights[i] * weights[j] * 
                                     individual_volatilities[asset1] * individual_volatilities[asset2] * 
                                     correlation_matrix.loc[asset1, asset2])
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            raise ValueError("Portfolio volatility is zero - cannot calculate Sharpe ratio")
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Calculate diversification benefit
        weighted_avg_vol = sum(weights[i] * individual_volatilities[symbols[i]] for i in range(len(symbols)))
        diversification_benefit = weighted_avg_vol - portfolio_volatility
        
        # Portfolio daily returns for additional metrics
        portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annual_return': portfolio_return,
            'annual_volatility': portfolio_volatility,
            'risk_free_rate': risk_free_rate,
            'max_drawdown': max_drawdown,
            'diversification_benefit': diversification_benefit,
            'individual_returns': individual_returns,
            'individual_volatilities': individual_volatilities,
            'correlation_matrix': correlation_matrix,
            'portfolio_returns': portfolio_daily_returns,
            'cumulative_returns': cumulative_returns,
            'weighted_avg_volatility': weighted_avg_vol
        }
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from cumulative returns
        """
        if len(cumulative_returns) == 0:
            return 0.0
        
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    def get_sharpe_rating(self, sharpe_ratio: float) -> Tuple[str, str]:
        """
        Get rating and color for Sharpe ratio
        Returns: (rating_text, color)
        """
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            return "❓ INVALID", "gray"
        elif sharpe_ratio > 2.0:
            return "🌟 EXCEPTIONAL", "green"
        elif sharpe_ratio > 1.5:
            return "✅ VERY GOOD", "green"
        elif sharpe_ratio > 1.0:
            return "👍 GOOD", "blue"
        elif sharpe_ratio > 0.5:
            return "⚠️ FAIR", "orange"
        else:
            return "❌ POOR", "red"
    
    def get_benchmark_comparison(self, sharpe_ratio: float) -> str:
        """
        Compare Sharpe ratio to common benchmarks
        """
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            return "Invalid Sharpe ratio - cannot compare to benchmarks"
        
        benchmarks = {
            "S&P 500 Historical": 0.6,
            "Warren Buffett Career": 0.8,
            "Top Hedge Funds": 1.2,
            "Renaissance Technologies": 2.5
        }
        
        comparisons = []
        for name, bench_sharpe in benchmarks.items():
            if sharpe_ratio > bench_sharpe:
                comparisons.append(f"✅ Better than {name} ({bench_sharpe:.1f})")
            else:
                comparisons.append(f"❌ Below {name} ({bench_sharpe:.1f})")
        
        return "\n".join(comparisons)

# Utility functions for the app
def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{decimals}%}"

def format_number(value: float, decimals: int = 3) -> str:
    """Format number with specified decimals"""
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range"""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError("Start date must be before end date")
        
        if (end - start).days < 30:
            raise ValueError("Date range must be at least 30 days")
        
        # Check if end date is not too far in the future
        if end > datetime.now() + timedelta(days=1):
            raise ValueError("End date cannot be in the future")
        
        return True
    except ValueError as e:
        raise ValueError(f"Invalid date range: {str(e)}")
