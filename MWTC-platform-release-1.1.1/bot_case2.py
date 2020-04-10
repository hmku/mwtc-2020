import grpc
import argparse
import threading
import time

import xchange.protos.exchange_pb2_grpc as exchange_grpc

import xchange.protos.competitor_registry_pb2 as comp_reg
import xchange.protos.order_book_pb2 as order_book
import xchange.protos.competitor_pb2 as competitor
import xchange.protos.data_feed_pb2 as data_feed

from xchange.client import Client

from xchange.client_util import OrderSide, OrderType
from xchange.competitor_bot import CompetitorBot

import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import scipy.stats as si
from statistics import stdev

TMIN = 10e-4

DATA_PATH = 'data/normalized_price_paths/history_0.csv'
NORMAL_CDF_PATH = 'case2/normal_cdf.csv'

class OptionBot(CompetitorBot):

    def __init__(self, *args, **kwargs):
        Client.__init__(self)
        self.num_assets = 10
        self.update_frequency = 8
        self.num_price_updates = 0
        self.total_price_updates = 450
        self.annual_price_updates = 1800

        self.delta_limit = 2000
        self.gamma_limit = 5000
        self.theta_limit = 5000
        self.vega_limit = 10e5

        self.underlyings = [chr(ord("A") + i) for i in range(self.num_assets)]
        self.strikes = list(range(70, 160, 10)) #50-205
        self.chains = {}
        self.all_assets = []
        for und in self.underlyings:
            opt_names = [und] + ["{}{}C".format(und, strike) for strike in self.strikes] + ["{}{}P".format(und, strike) for strike in self.strikes]
            self.chains[und] = pd.DataFrame(columns = ["theo", "bbid_px", "bbid_sz", "my_bid_px", "my_bid_sz", "my_bid_id", "bask_px", "bask_sz", "my_ask_px", "my_ask_sz", "my_ask_id", "delta", "gamma", "theta", "vega"], index = opt_names)
            self.chains[und]["position"] = 0
            self.chains[und]["my_bid_id"] = ""
            self.chains[und]["my_ask_id"] = ""
            self.chains[und].loc[und, "delta"] = 1
            self.chains[und].loc[und, "gamma"] = 0
            self.chains[und].loc[und, "theta"] = 0
            self.chains[und].loc[und, "vega"] = 0
            self.all_assets += opt_names
        self.all_assets = sorted(self.all_assets, key=len) #sorts assets from shortest to longest.  Will put underlyings first

        self.prices = pd.read_csv(DATA_PATH, index_col=0, header=0) # df of prices
        self.returns = self.prices.transpose().pct_change().dropna().transpose() # df of returns
        self.prices = self.prices.to_numpy().tolist() # convert to list of lists
        self.returns = self.returns.to_numpy().tolist() # convert to list of lists

        self.cdf_table = pd.read_csv(NORMAL_CDF_PATH, index_col=0, names='p').iloc[:,0] # cdf table
    
    def newton_vol_call(self, S, K, T, C, r, sigma):
    
        #S: spot price
        #K: strike price
        #T: time to maturity
        #C: Call value
        #r: interest rate
        #sigma: volatility of underlying asset
        
        d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
        
        vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
        
        tolerance = 0.000001
        x0 = sigma
        xnew  = x0
        xold = x0 - 1
            
        while abs(xnew - xold) > tolerance:
        
            xold = xnew
            xnew = (xnew - fx - C) / vega
            
        return abs(xnew)
    

    def newton_vol_put(self, S, K, T, P, r, sigma):
    
        d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
        
        vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
        
        tolerance = 0.000001
        x0 = sigma
        xnew  = x0
        xold = x0 - 1
            
        while abs(xnew - xold) > tolerance:
        
            xold = xnew
            xnew = (xnew - fx - P) / vega
            
        return abs(xnew)

    # helper function for finding rolling historical vol given lookback
    def rolling_vol(self, tx, lookback):
        i = ord(tx) - ord("A")
        return stdev(self.returns[i][len(self.returns[i])-lookback:])*math.sqrt(self.annual_price_updates)

    def gen_vol_level(self, tx, min_lookback=5, lookback_ratio=6):
        #come up with a vol estimate here
        return self.rolling_vol(tx, max(min_lookback, (self.total_price_updates - self.num_price_updates) // lookback_ratio))

    def option_pricer(self, cp_flag, S, K, T, v, r=0, q=0.0):
    
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #v: volatility of underlying asset
        
        spots = np.full(18, S)
        cp = np.concatenate([np.full(9, 1), np.full(9, -1)])
        d1 = ((np.log(spots / K) + (r + 0.5 * v * v) * T) / (v * np.sqrt(T))) * cp
        d2 = ((np.log(spots / K) + (r - 0.5 * v * v) * T) / (v * np.sqrt(T))) * cp
        d1_ind = np.clip((d1 + 4) * 1000, 0, 7999).astype(int)
        d2_ind = np.clip((d2 + 4) * 1000, 0, 7999).astype(int)
        cdfd1 = self.cdf_table[d1_ind].to_numpy()
        cdfd2 = self.cdf_table[d2_ind].to_numpy()
        result = (S * cdfd1 - K * cdfd2) * cp
        return result

    def option_delta(self, cp_flag, S, K, T, v, r = 0):
        return (self.option_pricer(cp_flag, S + TMIN, K, T, v, r) - self.option_pricer(cp_flag, S, K, T, v, r)) / TMIN

    def option_vega(self, cp_flag, S, K, T, v, r = 0):
        return (self.option_pricer(cp_flag, S, K, T, v + TMIN, r) - self.option_pricer(cp_flag, S, K, T, v, r)) / TMIN

    def option_theta(self, cp_flag, S, K, T, v, r = 0):
        return ((self.option_pricer(cp_flag, S, K, T + TMIN, v, r) - self.option_pricer(cp_flag, S, K, T, v, r)) / TMIN) / 252.

    def option_gamma(self, cp_flag, S, K, T, v, r = 0):
        return (self.option_delta(cp_flag, S + TMIN, K, T, v, r) - self.option_delta(cp_flag, S, K, T, v, r)) / TMIN

    def cancel_place(self, row):
        if len(row.name) == 1:
            return "", ""
        self.cancel_order(row["my_bid_id"], row.name)
        self.cancel_order(row["my_ask_id"], row.name)
        return self.place_order(OrderType.LIMIT, OrderSide.BID, row["my_bid_sz"], row.name, "%.2f" % round(row["my_bid_px"], 2))[1].order_id if row["my_bid_sz"] > 0 else "", self.place_order(OrderType.LIMIT, OrderSide.ASK, row["my_ask_sz"], row.name, "%.2f" % round(row["my_ask_px"], 2))[1].order_id if row["my_ask_sz"] > 0 else ""

    def handle_market_update(self, exchange_update_response):
        update = getattr(exchange_update_response, 'market_update')
        
        # append prices to prices array, update returns array
        for i, tx in enumerate(self.chains):
            try:
                curr_price = (float(update.book_updates[tx].bids[0].px) + float(update.book_updates[tx].asks[0].px)) / 2
            except IndexError:
                curr_price = float("inf")
            self.prices[i].append(curr_price)
            self.returns[i].append(self.prices[i][-1] / self.prices[i][-2] - 1)

        for underlying in self.chains:
            bbid_pxs = []
            bbid_szs = []
            bask_pxs = []
            bask_szs = []
            for a in self.chains[underlying].index:
                try:
                    bbid_px = float(update.book_updates[a].bids[0].px)
                    bbid_sz = float(update.book_updates[a].bids[0].qty)
                except IndexError:
                    bbid_px = 0
                    bbid_sz = 0
                try:
                    bask_px = float(update.book_updates[a].asks[0].px)
                    bask_sz = float(update.book_updates[a].asks[0].qty)
                except IndexError:
                    bask_px = float("inf")
                    bask_sz = 0
                bbid_pxs.append(bbid_px)
                bbid_szs.append(bbid_sz)
                bask_pxs.append(bask_px)
                bask_szs.append(bask_sz)

            self.chains[underlying]["bbid_px"] = bbid_pxs
            self.chains[underlying]["bbid_sz"] = bbid_szs
            self.chains[underlying]["bask_px"] = bask_pxs
            self.chains[underlying]["bask_sz"] = bask_szs

            is_option = self.chains[underlying].index.str.len() > 1
            options_only_chain = self.chains[underlying].loc[is_option]
            v = self.gen_vol_level(underlying)
            cp_flag = np.array(options_only_chain.index.str[-1])
            T = (self.total_price_updates - self.num_price_updates) / self.annual_price_updates
            K = np.array(options_only_chain.index.str[1:-1]).astype(int)
            S = (self.chains[underlying].loc[underlying, "bbid_px"] + self.chains[underlying].loc[underlying, "bask_px"]) / 2
            if math.isinf(S):
                break
            
            #print(underlying, v, cp_flag, T, K, S)

            theo = self.option_pricer(cp_flag, S, K, T, v)
            delta = self.option_delta(cp_flag, S, K, T, v)
            gamma = self.option_gamma(cp_flag, S, K, T, v)
            theta = self.option_theta(cp_flag, S, K, T, v)
            vega = self.option_vega(cp_flag, S, K, T, v)

            if underlying == 'A':
                print("Price of A: " + str(S))

            self.chains[underlying].loc[is_option, "theo"] = theo
            self.chains[underlying].loc[is_option, "delta"] = delta
            self.chains[underlying].loc[is_option, "gamma"] = gamma
            self.chains[underlying].loc[is_option, "theta"] = theta
            self.chains[underlying].loc[is_option, "vega"] = vega

            self.chains[underlying].loc[is_option, "my_ask_px"] = self.chains[underlying].loc[is_option, "theo"] + .01
            self.chains[underlying].loc[is_option, "my_bid_px"] = np.maximum(0, self.chains[underlying].loc[is_option, "theo"] - .01)
            self.chains[underlying].loc[is_option, "my_ask_sz"] = 1000
            self.chains[underlying].loc[is_option, "my_bid_sz"] = 1000

            asset_delta = (self.chains[underlying]["delta"] * self.chains[underlying]["position"]).sum()
            asset_gamma = (self.chains[underlying]["gamma"] * self.chains[underlying]["position"]).sum()
            asset_theta = (self.chains[underlying]["theta"] * self.chains[underlying]["position"]).sum()
            asset_vega = (self.chains[underlying]["vega"] * self.chains[underlying]["position"]).sum()

            b_a = pd.DataFrame(self.chains[underlying].apply(lambda row: self.cancel_place(row), axis = 1).tolist(), index = self.chains[underlying].index)
            b_a.columns = ["my_bid_id", "my_ask_id"]

            self.chains[underlying].drop(columns = ["my_bid_id", "my_ask_id"], inplace = True)
            self.chains[underlying] = self.chains[underlying].merge(b_a, left_index = True, right_index = True)

    def handle_fill_update(self, exchange_update_response):
        #therre are almost certainly faster ways to do this
        f = exchange_update_response.fill_update
        c = f.asset
        q = f.filled_qty
        rq = f.remaining_qty
        o = f.order_id
        sold = f.order_side == data_feed.FillUpdate.Side.Value('SELL')
        if sold:
            q *= -1

        self.chains[c[0]].loc[c, "position"] += q

        if q > 0:
            self.chains[c[0]].loc[c, "my_bid_sz"] -= q
            if self.chains[c[0]].loc[c, "my_bid_sz"] == 0:
                self.chains[c[0]].loc[c, "my_bid_id"] = ""
        else:
            self.chains[c[0]].loc[c, "my_ask_sz"] += q
            if self.chains[c[0]].loc[c, "my_ask_sz"] == 0:
                self.chains[c[0]].loc[c, "my_ask_id"] = ""

    def handle_exchange_update(self, exchange_update_response):
        if exchange_update_response.HasField('market_update') and float(exchange_update_response.market_update.book_updates["A"].bids[0].px) != float(self.chains["A"].loc["A", "bbid_px"]): #if its a market update and the its price change update
            self.num_price_updates += 1
            self.handle_market_update(exchange_update_response)
        if exchange_update_response.HasField('fill_update'):
            self.handle_fill_update(exchange_update_response)
        # for field in ['order_status_response','competition_event','pnl_update', 'liquidation_event']:
        for field in ['competition_event','pnl_update', 'liquidation_event']:
            try:
                if exchange_update_response.HasField(field):
                    print(field, getattr(exchange_update_response, field))
                    x = 0
            except:
                print("error with " + field)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="9090")
    parser.add_argument("--username", type=str)
    parser.add_argument("--password", type=str)

    args = parser.parse_args()

    client = OptionBot()
    client.start(args.host, args.port, args.username, args.password)
