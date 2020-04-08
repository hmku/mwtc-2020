import numpy as np
import pandas as pd
import scipy

#########################################################################
## Change this code to take in stock data and allocate your portfolio. ##
#########################################################################

market_betas = [1.2605551453269928, 1.6103656872028644, 0.7181943325951229, 2.05767007669806, 1.3312209339367767, 1.0704677322071188, 1.1819496118363249, 1.9134156627572563, 1.6082749182481006, 1.161738910693915, 1.0]
market_corrs = [0.9162626520817875, 0.9375055104218573, 0.7569115956488784, 0.9682450065023641, 0.9185611323436597, 0.8516020357611426, 0.91219262034569, 0.9481929321552719, 0.9552685262656853, 0.8762227363632008, 1.0]
prices = list()
holdings = np.array([0 for i in range(10)])
portfolio = []
market = 10
rate = 11
cash = 10000
weights = None
value = cash
def allocate_portfolio(stock_prices, market_price, risk_free_rate):
    global market_betas, prices, holdings, portfolio, market, rate, cash, weights, value

    # update matrix of historical prices
    prices.append(list(stock_prices) + [market_price] + [risk_free_rate])

    # determine weights
    if len(prices) <= 1:
        weights = np.repeat(0, 10)
    else:
        daily_returns = np.log(np.array(prices[-1]) / np.array(prices[-2]))
        new_weights = np.array([(daily_returns[i] - daily_returns[market] * market_betas[i]) * market_corrs[i] for i in range(10)])
        new_weights = new_weights / np.sum(np.abs(new_weights))
        weights = weights + 5 * risk_free_rate * new_weights
        weights = weights / np.sum(np.abs(weights))
    
    # update portfolio value and holdings
    new_holdings = np.multiply(weights, value) / stock_prices
    cash += np.sum((holdings - new_holdings) * stock_prices)
    holdings = new_holdings
    value = np.sum(holdings * stock_prices) + cash
    portfolio.append(list(holdings) + [cash, value, risk_free_rate])

    return weights