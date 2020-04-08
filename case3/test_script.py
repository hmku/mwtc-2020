import allocate
from allocate import allocate_portfolio
import numpy as np
import pandas as pd
import scipy
import math

dfr_raw = pd.read_csv("risk_free_train.csv", index_col=0)
dfr_test = dfr_raw[60:].reset_index()
del dfr_test['index']
dfp_raw = pd.read_csv("stock_prices_train.csv", index_col=0)
dfp_test = dfp_raw[1260:].reset_index()
del dfp_test['index']

holdings = np.array([0 for i in range(10)])
portfolio = []
cash = 10000
value = cash
for i in range(1260):
    stock_prices, market_price, risk_free_rate = np.array(dfp_test.iloc[i][:-1]), dfp_test.iloc[i][-1], dfr_test.iloc[i//21][0]
    weights = allocate_portfolio(stock_prices, market_price, risk_free_rate)
    
    # update portfolio value and holdings
    new_holdings = np.multiply(weights, value) / stock_prices
    cash += np.sum((holdings - new_holdings) * stock_prices)
    holdings = new_holdings
    value = np.sum(holdings * stock_prices) + cash
    portfolio.append(list(holdings) + [cash, value, risk_free_rate])

res = pd.DataFrame(portfolio, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'cash', 'value', 'rate'])
der = res['value'].pct_change() - res['rate'][1:] / 252
sharpe = math.sqrt(252) * der.mean() / der.std()
print(sharpe)