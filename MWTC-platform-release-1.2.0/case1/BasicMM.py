#!/usr/bin/env python
# coding: utf-8

import math

class BasicMM():
    
    def __init__(self, tick_sizes):
        '''
        Maximum Range: The maximum range you think the price could move in; i.e. the highest 
        possible price minus the lowest
        Tick Sizes: Minimum price fluctuation of the contract
        '''
        
        self.contracts = list(tick_sizes.keys())
        self.pos = {c: 0 for c in self.contracts}
        
        # self.max_ranges = max_ranges #{c: 100 for c in self.contracts}
        self.tick_sizes = tick_sizes #{c: 0.01 for c in self.contracts}
        
    def basic_mm(self, ltf, width, max_pos, clip, c, max_ranges):
        '''
        Long-term fair (ltf): your best guess at the value
        Width: How wide you quote; i.e. the difference between your bid and ask
        Maximum position (max_pos): The maximum number of contracts you are willing to hold
        Maximum Clip: The number of contracts you quote on each level
        c: contract identifies

        A basic function takes the above parameter and returns a dictionary of information
        you can use to market make while also incorporating information you receive from the
        fill you get from the market.
        '''

        #Penalty (pen): How much you adjust your estiamte of the price given you get
        #filled for one lot
        #Optimized so that you reach your max position exactly when the max range is reached
        pen = (max_ranges[c] / 2.0) / max_pos
        #Short-Term Fair (stf): Your temporary fair incorporating the fills you receive
        stf = ltf - self.pos[c] * pen

        #your best bid and ask prices
        #remember prices have to be in increments of tick_size
        bp_1 = self.round_to_inc(stf - width / 2.0, self.tick_sizes[c], False)
        ap_1 = self.round_to_inc(stf + width / 2.0, self.tick_sizes[c])

        #your next best bid and ask
        bp_2 = min(self.round_to_inc(stf - clip * pen - width / 2.0, 
                   self.tick_sizes[c], False), bp_1 - self.tick_sizes[c])
        ap_2 = max(self.round_to_inc(stf + clip * pen + width / 2.0,
                   self.tick_sizes[c]), ap_1 + self.tick_sizes[c])

        # really wide orders
        bp_3 = self.round_to_inc(stf - width * 3.0, self.tick_sizes[c], False)
        ap_3 = self.round_to_inc(stf + width * 3.0, self.tick_sizes[c])

        # even wider orders
        bp_4 = self.round_to_inc(stf - 0.4, self.tick_sizes[c], False)
        ap_4 = self.round_to_inc(stf + 0.4, self.tick_sizes[c])

        #position management:
        # bids_left = max_pos - self.pos[c]
        # asks_left = max_pos + self.pos[c]
        # if bids_left <= 0:
        #     #puke your position!
        #     ap_1 = bp_1
        #     as_1 = clip
        #     ap_2 = bp_1 + self.tick_sizes[c]
        #     as_2 = clip
        #     as_3 = 100
        #     bs_1 = 0
        #     bs_2 = 0
        #     bs_3 = 0
        # elif asks_left <= 0:
        #     #puke your position!
        #     bp_1 = ap_1
        #     bs_1 = clip
        #     bp_2 = ap_1 - self.tick_sizes[c]
        #     bs_2 = clip
        #     bs_3 = 100
        #     as_1 = 0
        #     as_2 = 0
        #     as_3 = 0
        # else:
        #     #bid and ask size setting
        #     bs_1 = min(bids_left, clip // 2)
        #     bs_2 = max(0, min(bids_left - clip, clip))
        #     as_1 = min(asks_left, clip // 2)
        #     as_2 = max(0, min(asks_left - clip, clip))
        #     bs_3 = 100
        #     as_3 = 100

        # position management
        bs_1 = int(-clip * self.pos[c] / max_pos + clip)
        as_1 = int(clip * self.pos[c] / max_pos + clip)
        bs_2 = min(bs_1 * 2, 100)
        as_2 = min(as_1 * 2, 100)
        bs_3 = max(min(100, (max_pos - self.pos[c] - bs_1 - bs_2)), 0)
        as_3 = max(min(100, (max_pos + self.pos[c] - as_1 - as_2)), 0)
        bs_4 = max(min(100, (max_pos - self.pos[c] - bs_1 - bs_2 - bs_3)), 0)
        as_4 = max(min(100, (max_pos + self.pos[c] - as_1 - as_2 - as_3)), 0)

        return {'contract': c,
                'bid_prices': [bp_1, bp_2, bp_3, bp_4], 
                'bid_sizes': [bs_1, bs_2, bs_3, bs_4],
                'ask_prices': [ap_1, ap_2, ap_3, ap_4],
                'ask_sizes': [as_1, as_2, as_3, as_4],
                'stf': stf,
                'pen': pen}
    
    def update_pos(self, c, change):
        '''
        updates position
        '''
        self.pos[c] = self.pos[c] + change
    
    def round_to_inc(self, num, increment, ask=True):
        '''
        Rounds a bid or ask to the correct price increment
        '''
        mult = num / increment
        if ask:
            mult = math.ceil(mult)
        else: 
            mult = math.floor(mult)
        return round(increment * mult, 10)

