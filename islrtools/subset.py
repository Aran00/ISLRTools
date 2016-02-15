''' The traditional subset chosen method '''
__author__ = 'Aran'

import sys
import numpy as np
import statsmodels.formula.api as smf
from itertools import combinations
from pandas import DataFrame, Series

# stat_columns = ['adjust R^2', 'AIC', 'BIC', 'RSS']

'''
formula: The regression formula
data: DataFrame
nvmax: The largest subset feature count
method: combination/forward/backward
'''
def regsubsets(formula, data, nvmax, method=None):
    if method == 'forward':
        return regsubsets_forward(formula, data, nvmax)
    elif method == 'backward':
        return regsubsets_backward(formula, data, nvmax)
    else:
        return regsubsets_comb(formula, data, nvmax)

def regsubsets_comb(formula, data, nvmax):
    x_col_list, y_col, output, stat = _parse_formula_data(formula, data, nvmax)
    for i in xrange(min(len(x_col_list), nvmax)):
        min_rss = sys.maxsize
        chosen_subset = None
        for sub_x_cols in combinations(x_col_list, i+1):
            sub_formula = "%s~%s" % (y_col, "+".join(sub_x_cols))
            model = smf.ols(formula=sub_formula, data=data)
            result = model.fit()
            if result.ssr < min_rss:
                min_rss = result.ssr
                chosen_subset = sub_x_cols
                stat.ix[i+1] = result
        for x_col in chosen_subset:
            output.ix[i+1][x_col] = True
    return output, stat

def regsubsets_forward(formula, data, nvmax):
    x_col_list, y_col, output, stat = _parse_formula_data(formula, data, nvmax)
    chosen_subset = []
    sub_formula = "%s~" % y_col
    for i in xrange(min(len(x_col_list), nvmax)):
        min_rss = sys.maxsize
        chosen_x_col = None
        for x_col in x_col_list:
            temp_formula = "%s+%s" % (sub_formula, x_col)
            model = smf.ols(formula=temp_formula, data=data)
            result = model.fit()
            if result.ssr < min_rss:
                min_rss = result.ssr
                chosen_x_col = x_col
                stat.ix[i+1] = result
        chosen_subset.append(chosen_x_col)
        x_col_list.remove(chosen_x_col)
        sub_formula = "%s+%s" % (sub_formula, chosen_x_col)
        for col in chosen_subset:
            output.ix[i+1][col] = True
    return output, stat

def regsubsets_backward(formula, data, nvmax):
    x_col_list, y_col, output, stat = _parse_formula_data(formula, data, nvmax, True)
    chosen_subset = np.array(x_col_list)
    col_size = len(x_col_list)
    # The params for the whole model
    sub_formula = "%s~%s" % (y_col, "+".join(chosen_subset))
    model = smf.ols(formula=sub_formula, data=data)
    result = model.fit()
    output.ix[col_size] = True
    stat[col_size] = result
    for i in xrange(1, min(col_size, nvmax)):
        min_rss = sys.maxsize
        chosen_x_col_index = -1
        for j in xrange(len(chosen_subset)):
            remain_subset = np.delete(chosen_subset, j)
            sub_formula = "%s~%s" % (y_col, "+".join(remain_subset))
            model = smf.ols(formula=sub_formula, data=data)
            result = model.fit()
            if result.ssr < min_rss:
                min_rss = result.ssr
                chosen_x_col_index = j
                stat[col_size - i] = result
        chosen_subset = np.delete(chosen_subset, chosen_x_col_index)
        for col in chosen_subset:
            output.ix[col_size - i][col] = True
    return output, stat

def _parse_formula_data(formula, data, nvmax, backward=False):
    split_symbol_index = formula.index("~")
    y_col = formula[:split_symbol_index]
    x_cols = formula[(split_symbol_index+1):]
    if x_cols == ".":
        x_col_list = data.columns.tolist()
        x_col_list.remove(y_col)
    else:
        x_col_list = x_cols.split("+")
    col_count = len(x_col_list)
    row_count = min(col_count, nvmax)
    index = range(1, row_count+1) if not backward else range(col_count-row_count+1, col_count+1)
    initial_output = DataFrame(np.zeros((row_count, col_count), dtype=bool), index=index, columns=x_col_list)
    statistics = Series(np.zeros(row_count), index=index, name='ols_result')
    return x_col_list, y_col, initial_output, statistics

'''
def _get_statistics(fit_result):
    return np.array([fit_result.rsquared_adj, fit_result.aic, fit_result.bic, fit_result.ssr])
'''