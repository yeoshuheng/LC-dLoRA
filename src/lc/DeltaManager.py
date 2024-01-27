from src.Config import Config
import numpy as np
import math
from collections import defaultdict

class DeltaManager:

    def __init__(self, config : Config, base_tuple : tuple):
        self.bit_width  = config.lc_bw
        self.full_delta_base, self.decomposed_delta_base = base_tuple
    
    def take_delta(self, current_state_tuple : tuple) -> tuple:
        full_current_state, decomposed_current_state = current_state_tuple
        delta_full_state = np.subtract(full_current_state, 
                                       self.full_delta_base)
        delta_decomposed_state = np.subtract(decomposed_current_state, 
                                             self.decomposed_delta_base)
        promoted_full_delta = self.priority_promotion(delta_full_state)
        promoted_decomposed_delta = self.priority_promotion(delta_decomposed_state)
        self.full_delta_base = np.add(self.full_delta_base, 
                                      promoted_full_delta)
        self.decomposed_delta_base = np.add(self.decomposed_delta_base, 
                                            promoted_decomposed_delta)
        return (promoted_full_delta, promoted_decomposed_delta)

    def priority_promotion(self, δt: np.array) -> np.array:
        _, δt_exp = np.frexp(δt)
        δt_sign = np.sign(δt)
        δt_sign[δt_sign > 0] = 0
        δt_sign[δt_sign < 0] = 1    
        mp =  defaultdict(list)
        for i in range(len(δt)):
            mp[(δt_exp[i], δt_sign[i])].append((i, δt[i]))
        for k in mp:
            mp[k] = (np.average(np.array([x[-1] for x in mp[k]])), 
                    [x[0] for x in mp[k]])
        mp = list(mp.values())
        allowed_buckets = int(math.pow(2, self.bit_width) - 1)
        mp = sorted(mp, key = lambda x : abs(x[0]), reverse = True)[:min(allowed_buckets, len(mp))]
        new_δt= [0 for x in range(len(δt))]
        for qtVal, pos in mp:
            for p in pos:
                new_δt[p] = qtVal
        new_δt = np.array(new_δt, dtype = np.float32) # must ensure float32 @ save for proper buffer read.