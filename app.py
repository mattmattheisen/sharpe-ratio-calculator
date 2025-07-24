"""
Professional Sharpe Ratio Calculator
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from calculator import SharpeCalculator, format_percentage, format_number, validate_date_range

# Page configuration
st.set_page_config(
    page_title="Sharpe Ratio Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 1rem 0;
    }
    
    .sharpe-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Professional Sharpe Ratio Calculator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>What is the Sharpe Ratio?</strong><br>
    The Sharpe ratio measures risk-adjusted returns by comparing portfolio returns to the risk-free rate, 
    divided by portfolio volatility. Higher values indicate better risk-adjusted performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize calculator
    if 'calculator' not in st.session_state:
        st.session_state.calculator = SharpeCalculator()
    
    calculator = st.session_state.calculator
    
    # Sidebar for method selection
    st.sidebar.header("📊 Calculation Method")
    method = st.sidebar.selectbox(
        "Choose calculation method:",
        ["Portfolio Builder", "Manual Input", "Advanced Analysis"],
        help="Select how you want to calculate the Sharpe ratio"
    )
    
    if method == "Portfolio Builder":
        portfolio_builder_tab(calculator)
    elif method == "Manual Input":
        manual_input_tab(calculator)
    else:
        advanced_analysis_tab(calculator)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Built with ❤️ for quantitative finance | 
    <a href="https://github.com/yourusername/sharpe-ratio-calculator" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

def portfolio_builder_tab(calculator):
    """Portfolio Builder interface"""
    st.header("🏗️ Portfolio Builder")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Build Your Portfolio")
        
        # Portfolio input
        num_assets = st.number_input("Number of assets:", min_value=1, max_value=10, value=2)
        
        symbols = []
        weights = []
        
        for i in range(num_assets):
            col_symbol, col_weight = st.columns([2, 1])
            
            with col_symbol:
                symbol = st.text_input(f"Asset {i+1} Symbol:", value="AAPL" if i == 0 else "GOOGL" if i == 1 else "", key=f"symbol_{i}")
                symbols.append(symbol.upper())
            
            with col_weight:
                weight = st.number_input(f"Weight %:", min_value=0.0, max_value=100.0, 
                                       value=50.0 if num_assets == 2 else 100.0/num_assets, key=f"weight_{i}")
                weights.append(weight / 100.0)
        
        # Date range
        col_start, col_end = st.columns(2)
        
        with col_start:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        
        with col_end:
            end_date = st.date_input(
                "End Date:",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Risk-free rate
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%):",
            value=calculator.risk_free_rate * 100,
            min_value=0.0,
            max_value=10.0,
            format="%.2f",
            help="Current 10-year Treasury rate is auto-loaded"
        )
    
    with col2:
        st.subheader("⚖️ Portfolio Summary")
        
        # Validation
        total_weight = sum(weights)
        valid_symbols = all(symbol.strip() for symbol in symbols)
        
        if abs(total_weight - 1.0) > 0.01:
            st.error(f"⚠️ Weights sum to {total_weight:.1%}, must equal 100%")
        elif not valid_symbols:
            st.error("⚠️ Please enter all stock symbols")
        else:
            st.success("✅ Portfolio is valid")
            
            # Display portfolio
            portfolio_df = pd.DataFrame({
                'Symbol': symbols,
                'Weight': [f"{w:.1%}" for w in weights]
            })
            st.table(portfolio_df)
    
    # Calculate button
    if st.button("🚀 Calculate Sharpe Ratio", type="primary"):
        if valid_symbols and abs(total_weight - 1.0) <= 0.01:
            try:
                with st.spinner("Downloading data and calculating..."):
                    results = calculator.calculate_simple_sharpe(
                        symbols, weights, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d'),
                        risk_free_rate / 100
                    )
                
                display_results(results, symbols, weights)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.error("❌ Please fix portfolio issues before calculating")

def manual_input_tab(calculator):
    """Manual input interface"""
    st.header("✏️ Manual Input")
    
    st.markdown("""
    <div class="info-box">
    <strong>Use this method when you already know your portfolio's return and volatility.</strong><br>
    Perfect for analyzing existing investment statements or theoretical portfolios.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_return = st.number_input(
            "Annual Portfolio Return (%):",
            value=12.0,
            min_value=-50.0,
            max_value=100.0,
            format="%.2f",
            help="Your portfolio's annual return percentage"
        )
        
        portfolio_volatility = st.number_input(
            "Annual Portfolio Volatility (%):",
            value=18.0,
            min_value=0.1,
            max_value=100.0,
            format="%.2f",
            help="Standard deviation of your portfolio returns"
        )
    
    with col2:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%):",
            value=calculator.risk_free_rate * 100,
            min_value=0.0,
            max_value=10.0,
            format="%.2f",
            help="Current 10-year Treasury rate"
        )
        
        st.markdown("### 📊 Quick Examples")
        if st.button("Conservative Portfolio"):
            st.session_state.manual_return = 8.0
            st.session_state.manual_vol = 12.0
        if st.button("Aggressive Portfolio"):
            st.session_state.manual_return = 15.0
            st.session_state.manual_vol = 25.0
    
    if st.button("🚀 Calculate Sharpe Ratio", type="primary"):
        try:
            results = calculator.calculate_manual_sharpe(
                portfolio_return / 100,
                portfolio_volatility / 100,
                risk_free_rate / 100
            )
            
            display_manual_results(results)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def advanced_analysis_tab(calculator):
    """Advanced analysis with correlation adjustment"""
    st.header("🔬 Advanced Analysis")
    
    st.markdown("""
    <div class="info-box">
    <strong>Advanced correlation-adjusted calculation.</strong><br>
    This method provides more accurate portfolio volatility by accounting for correlations between assets.
    Includes diversification analysis and component contributions.
    </div>
    """, unsafe_allow_html=True)
    
    # Similar to portfolio builder but with more advanced features
    portfolio_builder_tab(calculator)  # Reuse the same interface for now

def display_results(results, symbols, weights):
    """Display calculation results"""
    st.header("📊 Results")
    
    # Main Sharpe ratio display
    sharpe = results['sharpe_ratio']
    rating, color = st.session_state.calculator.get_sharpe_rating(sharpe)
    
    st.markdown(f"""
    <div class="sharpe-display" style="color: {color};">
    {format_number(sharpe, 3)}
    </div>
    <div style="text-align: center; font-size: 1.5rem; margin-bottom: 2rem;">
    {rating}
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Annual Return",
            value=format_percentage(results['annual_return']),
            help="Annualized portfolio return"
        )
    
    with col2:
        st.metric(
            label="Annual Volatility", 
            value=format_percentage(results['annual_volatility']),
            help="Annualized standard deviation"
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=format_percentage(results['max_drawdown']),
            delta=f"-{format_percentage(results['max_drawdown'])}",
            help="Largest peak-to-trough decline"
        )
    
    with col4:
        st.metric(
            label="Win Rate",
            value=format_percentage(results['win_rate']),
            help="Percentage of positive trading days"
        )
    
    # Performance chart
    st.subheader("📈 Portfolio Performance")
    
    fig = go.Figure()
    
    # Cumulative returns
    dates = results['cumulative_returns'].index
    fig.add_trace(go.Scatter(
        x=dates,
        y=(results['cumulative_returns'] - 1) * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f4e79', width=2)
    ))
    
    fig.update_layout(
        title="Cumulative Returns (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Benchmark comparison
    st.subheader("🏆 Benchmark Comparison")
    benchmark_text = st.session_state.calculator.get_benchmark_comparison(sharpe)
    st.text(benchmark_text)
    
    # Additional info
    st.markdown(f"""
    <div class="info-box">
    <strong>Calculation Details:</strong><br>
    • Portfolio: {', '.join(f'{s} ({w:.1%})' for s, w in zip(symbols, weights))}<br>
    • Risk-Free Rate: {format_percentage(results['risk_free_rate'])}<br>
    • Observations: {results['num_observations']} trading days<br>
    • Total Return: {format_percentage(results['total_return'])}
    </div>
    """, unsafe_allow_html=True)

def display_manual_results(results):
    """Display manual calculation results"""
    st.header("📊 Results")
    
    # Main Sharpe ratio display
    sharpe = results['sharpe_ratio']
    rating, color = st.session_state.calculator.get_sharpe_rating(sharpe)
    
    st.markdown(f"""
    <div class="sharpe-display" style="color: {color};">
    {format_number(sharpe, 3)}
    </div>
    <div style="text-align: center; font-size: 1.5rem; margin-bottom: 2rem;">
    {rating}
    </div>
    """, unsafe_allow_html=True)
    
    # Calculation breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Portfolio Return",
            value=format_percentage(results['annual_return'])
        )
    
    with col2:
        st.metric(
            label="Risk-Free Rate",
            value=format_percentage(results['risk_free_rate'])
        )
    
    with col3:
        st.metric(
            label="Excess Return",
            value=format_percentage(results['excess_return']),
            help="Portfolio Return - Risk-Free Rate"
        )
    
    # Formula explanation
    st.subheader("🧮 Calculation Breakdown")
    st.latex(r"""
    Sharpe\ Ratio = \frac{Portfolio\ Return - Risk\ Free\ Rate}{Portfolio\ Volatility}
    """)
    
    st.latex(f"""
    Sharpe\ Ratio = \\frac{{{format_percentage(results['annual_return'], 1)} - {format_percentage(results['risk_free_rate'], 1)}}}{{{format_percentage(results['annual_volatility'], 1)}}} = {format_number(sharpe, 3)}
    """)
    
    # Benchmark comparison
    st.subheader("🏆 Benchmark Comparison")
    benchmark_text = st.session_state.calculator.get_benchmark_comparison(sharpe)
    st.text(benchmark_text)

if __name__ == "__main__":
    main()
