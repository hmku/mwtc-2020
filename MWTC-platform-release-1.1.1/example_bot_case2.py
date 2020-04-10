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

TMIN = 10e-4

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

    
    def newton_vol_call(S, K, T, C, r, sigma):
    
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
    

    def newton_vol_put(S, K, T, P, r, sigma):
    
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


    def gen_vol_level(self, other_params = None):
        #come up with a vol estimate here
        return .5
    
    def option_pricer(self, cp_flag, S, K, T, v, r=0, q=0.0):
    
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #v: volatility of underlying asset
        
        spots = np.full(18, S)
        cp = np.concatenate([np.full(9, 1), np.full(9, -1)])
        d1 = ((np.log(spots / K) + (r + 0.5 * v ** 2) * T) / (v * np.sqrt(T))) * cp
        d2 = ((np.log(spots / K) + (r - 0.5 * v ** 2) * T) / (v * np.sqrt(T))) * cp
        cdfd1 = np.array([si.norm.cdf(d, 0.0, 1.0) for d in d1])
        cdfd2 = np.array([si.norm.cdf(d, 0.0, 1.0) for d in d2])
        result = (S * cdfd1 - K * np.exp(-r * T) * cdfd2) * cp
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
            v = self.gen_vol_level()
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

        # if exchange_update_response.HasField('market_update'):
        #     mu = exchange_update_response.market_update
        #     print(mu)
        #     vol = 0
        #     strike = 0
        #     r = 0
        #     time = 0
        #     spot = 0
            # bu = exchange_update_response.market_update.book_updates
            # for option in bu:
            #     if len(bu[option].bids) > 0 and len(bu[option].asks) > 0:
            #         v = bu[option]
            #         fp = (v.bids[0].px * v.bids[0].qty + v.asks[0].px * v.asks[0].qty)/(v.bids[0].qty + v.asks[0].qty)
            #         self.option[option] = fp
            #         iv = 0
            #         print(fp)

            #         if len(option) > 2:
            #             strike = option[1:-1]
            #             r = 0                   # Risk free rate
            #             v = 0.5                 # volatility (adjusted)
            #             time = 0                # Time until expiration
            #             spot = 0                # Spot price
            #             C = fp                  # Call Value
            #             if option[-1] == 'C':
            #                 iv = newton_vol_call(spot, strike, time, C, r, v)
            #             else:
            #                 iv = newton_vol_put(spot, strike, time, C, r, v)
                
            #         self.iv[option] = iv

        if exchange_update_response.HasField('market_update') and float(exchange_update_response.market_update.book_updates["A"].bids[0].px) != float(self.chains["A"].loc["A", "bbid_px"]): #if its a market update and the its price change update
            self.num_price_updates += 1
            self.handle_market_update(exchange_update_response)
        if exchange_update_response.HasField('fill_update'):
            self.handle_fill_update(exchange_update_response)
        for field in ['order_status_response','competition_event','pnl_update', 'liquidation_event']:
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
