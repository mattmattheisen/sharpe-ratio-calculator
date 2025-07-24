def download_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download stock data with error handling - FIXED VERSION
    """
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        
        if len(symbols) == 1:
            # For single stock, yfinance returns different structure
            if isinstance(data.index, pd.DatetimeIndex):
                # Data downloaded successfully
                return data[['Adj Close']].rename(columns={'Adj Close': symbols[0]})
            else:
                raise ValueError(f"No data available for {symbols[0]}")
        else:
            # Multiple stocks
            return data['Adj Close']
            
    except Exception as e:
        raise ValueError(f"Error downloading data for {symbols}: {str(e)}")
