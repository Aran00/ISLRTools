__author__ = 'Aran'

''' R like bootstrap function '''

import numpy as np
from numpy import random as nprand

def boot(data, statistic, R):
    N = len(data)
    arr_results = []
    for i in xrange(R):
        index = nprand.choice(N, N, replace=True)
        arr_results.append(statistic(data, index))
    bs_results = np.array(arr_results)
    var_count = 1 if len(bs_results.shape) == 1 else bs_results.shape[1]
    if var_count == 1:
        bs_result = np.array([np.mean(bs_results), np.std(bs_results)])
    else:
        bs_result = np.zeros((var_count, 2))
        for i in xrange(var_count):
            bs_result[i] = (np.mean(bs_results[:, i]), np.std(bs_results[:, i]))
    return bs_result
