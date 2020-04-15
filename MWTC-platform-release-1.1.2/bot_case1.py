import grpc
import argparse
import threading
import time
import json
import pickle
from datetime import date, timedelta, datetime
from bisect import bisect_right

import xchange.protos.exchange_pb2_grpc as exchange_grpc

import xchange.protos.competitor_registry_pb2 as comp_reg
import xchange.protos.data_feed_pb2 as data_feed
import xchange.protos.order_book_pb2 as order_book
import xchange.protos.competitor_pb2 as competitor
from xchange.client_util import OrderSide, OrderType
from xchange.competitor_bot import CompetitorBot

from case1.BasicMM import BasicMM

#Parameters for BasicMM
MAX_RANGES = {'EFM':0.1, 'EFQ':0.1, 'EFV': 0.1, 'EFZ': 0.1} # changed this for now
TICK_SIZES = {'EFM':0.01, 'EFQ':0.01, 'EFV':0.01, 'EFZ':0.01}
#During the competition, LTFs should be dynamic: change this
# LTFS = {'EFM':2.5, 'EFQ':3.0, 'EFV': 3.5, 'EFZ': 4.0}

# import cdfs
with open('case1/cdf_file', 'rb') as cdf_file:
    CDFS = pickle.load(cdf_file)

# starting gas and coal amounts
GAS_AMTS = {(2019 + i): 1405.562093 + i * 11.729121446426436 for i in range(5)}
COAL_AMTS = {(2019 + i): 1904.282832 + i * 18.016372633113733 for i in range(5)}
COAL_REG = [1.475141438459995, -168.77216070111626] # slope, y-intercept of coal to gas regression

class SampleBot(CompetitorBot):
    def __init__(self):
        CompetitorBot.__init__(self)

        self.contracts = ['EF' + x for x in ['M','Q','V','Z']]
        self.expiries = dict(zip(self.contracts, [
            date(2000, 6, 30), date(2000, 8, 31), date(2000, 10, 30), date(2000, 12, 31)]))
        #This class does gives basic market making orders based on your position
        #See BasicMM.py
        self.mm = BasicMM(MAX_RANGES, TICK_SIZES)

        #These track your orders
        self.bids = {c: {} for c in self.contracts}
        self.asks = {c: {} for c in self.contracts}

        self.max_pos = 1000

        #can/should be dynamic, and be different for each contract.  change this.
        self.ltfs = {}
        self.clip = 100
        self.width = 0.10

        # gas/coal stores for this year (set to None for now)
        self.gas = None
        self.coal = None

    def handle_round_started(self):
        def f():
            '''
            Checks for contracts you're not quoting and quotes them
            '''
            while True:
                if self.creds is not None and len(self.ltfs) > 0: # only trade if we have fairs
                    for c in self.contracts:
                        if (self.bids[c] == {}) and (self.asks[c] == {}):
                            #Get orders from market maker
                            ords = self.mm.basic_mm(self.ltfs[c], self.width, self.max_pos, self.clip, c)
                            for i in range(min(len(ords['bid_prices']), len(ords['ask_prices']))):
                                o = self.place_order(OrderType.LIMIT, OrderSide.BID, ords['bid_sizes'][i], c, str(
                                    ords['bid_prices'][i]))[1]
                                self.bids[c][o.order_id] = {'q': ords['bid_sizes'][i], 'p': ords['bid_prices'][i]}
                                o = self.place_order(OrderType.LIMIT, OrderSide.ASK, ords['ask_sizes'][i], c, str(
                                    ords['ask_prices'][i]))[1]
                                self.asks[c][o.order_id] = {'q': ords['ask_sizes'][i], 'p': ords['ask_prices'][i]}

        threading.Thread(target=f).start()


    def handle_exchange_update(self, exchange_update_response):
        #Possible updates: 'market_update','fill_update','order_status_response','competition_event','pnl_update', etc.

        # if exchange_update_response.HasField('freeze_event'):
        #     print('Freeze!')
        #     print(exchange_update_response.freeze_event.message)

        if exchange_update_response.HasField('competition_event'):
            msg = json.loads(exchange_update_response.competition_event.message)
            print(msg) # date, consumption, avg_price, cons_change
            
            today = datetime.strptime(msg['DATE'], '%Y-%m-%d').date()
            cons, price, consch = msg['CONSUMPTION'], msg['AVG_PRICE'], msg['CONS_CHANGE']

            if self.gas is None or self.coal is None: # start of round, update year info
                self.gas = GAS_AMTS[today.year]
                self.coal = COAL_AMTS[today.year]
                for contract in self.expiries: # fix expiries
                    self.expiries[contract] = self.expiries[contract].replace(year=today.year)

            # when we run out of gas
            if price > 2 and price < 3:
                self.gas = cons - (price - 2) * consch
                self.coal = COAL_REG[0] * self.gas + COAL_REG[1]

            # update ltfs
            for contract, expiry in self.expiries.items():
                d = (expiry - today).days # days until expiry
                if d < 1 or d > 365:
                    continue # uh oh get out
                g = max(0, int(self.gas - cons) + 1) # gas left
                c = max(0, int(self.coal - cons) + 1) # coal left
                s = CDFS[0][d] # shift
                l = len(CDFS[d]) # range of values we have probabilities for
                pg = 0 if g - s < 0 else 1 if g - s >= l else CDFS[d][g-s] # p(gas at expiry)
                pc = 0 if c - s < 0 else 1 if c - s >= l else CDFS[d][c-s] # p(coal at expiry)
                pc = pc - pg # subtract the cumulatives to get the marginal probability
                pp = 1 - pc - pg # p(peakers at expiry)
                ppcons = bisect_right(CDFS[d], pc + pg + (pp/2)) - 1 # avg consumption given peaker plants
                self.ltfs[contract] = pg * 2 + pc * 3 + pp * (max(3, price) + (ppcons + s - c) * 0.001)
            
            print(self.ltfs)

        if exchange_update_response.HasField('market_update'):
            print(exchange_update_response.market_update)

        if exchange_update_response.HasField('pnl_update'):
            print(exchange_update_response.pnl_update)
        #Check for fills
        if exchange_update_response.HasField('fill_update'):
            cancel_list = []
            f = exchange_update_response.fill_update
            c = f.asset
            q = f.filled_qty
            rq = f.remaining_qty
            o = f.order_id
            side = 1
            if f.order_side == data_feed.FillUpdate.Side.SELL:
                side = -1
            #remove orders that have been filled from your order tracker
            #Update orders that have been partially filled
            if rq > 0:
                try:
                    if side == 1:
                        self.bids[c][o]['q'] = rq
                    else:
                        self.asks[c][o]['q'] = rq
                #check for order already filled
                except KeyError:
                    rq = 0
            if rq == 0:
                if side == 1:
                    self.bids[c].pop(o, None)
                else:
                    self.asks[c].pop(o, None)

            #Market maker needs to update stf based on new position
            self.mm.update_pos(c, side * q)
            #Get new orders from market maker
            ords = self.mm.basic_mm(self.ltfs[c], self.width, self.max_pos, self.clip, c)
            #Place any order changes to the exchange
            for ba in ['bid','ask']:
                ords[ba + '_prices'] = [x for i,x in enumerate(
                    ords[ba + '_prices']) if ords[ba + '_sizes'][i] != 0]
                ords[ba + '_sizes'] = [x for x in ords[ba + '_sizes'] if x != 0]
                no_cancel = []
                new_orders = {}
                #Iterate through new orders
                for i,x in enumerate(ords[ba + '_prices']):
                    y = ords[ba + '_sizes'][i]
                    done = False
                    for oid,v in getattr(self, ba + 's')[c].items():
                        #check if you already have orders that match your new order
                        if v['p'] == x:
                            done = True
                            if v['q'] > y:
                                new = self.modify_order(oid, OrderType.LIMIT, OrderSide.BID if (ba == 'bid') else OrderSide.ASK, y, c, str(x))[1]
                                new_orders[new.order_id] = {'q': y, 'p': x}
                            else:
                                no_cancel.append(oid)
                            break
                    #If not, place order
                    if not done:
                        new = self.place_order(OrderType.LIMIT, OrderSide.BID if (ba == 'bid') else OrderSide.ASK, y, c, str(x))[1]
                        new_orders[new.order_id] = {'q': y, 'p': x}
                for oid in list(getattr(self, ba + 's')[c].keys()):
                    #cancel orders that you no longer need
                    if oid not in no_cancel:
                        self.cancel_order(oid, c)
                        #remove cancelled orders from your tracker
                        getattr(self, ba + 's')[c].pop(oid, None)
                #Update your order tracker with new orders
                for k,v in new_orders.items():
                    getattr(self, ba + 's')[c][k] = v
            # if c == 'EFM':
            #     print(c, self.mm.pos[c], ords['stf'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the exchange client')
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="9090")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)

    args = parser.parse_args()
    client = SampleBot()
    client.start(args.host, args.port, args.username, args.password)
