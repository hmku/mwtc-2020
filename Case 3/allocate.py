import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in stock data and allocate your portfolio. ##
#########################################################################


def allocate_portfolio(stock_prices, market_price, risk_free_rate):
    
    # This simple strategy equally weights all stocks every period
    # (called a 1/n strategy).
    
    n_stocks = len(stock_prices)
    weights = np.repeat(1 / n_stocks, n_stocks)
    return weights
