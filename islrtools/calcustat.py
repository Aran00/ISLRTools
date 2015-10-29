__author__ = 'ryu'

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import Series
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_covariance(df):
    '''exercise-03-09(b)'''
    print np.cov(df, rowvar=0).shape
    columns = df.columns.tolist()
    cov_df = pd.DataFrame(np.corrcoef(df, rowvar=0), columns=columns, index=columns)
    print "The correlation coefficients of each column is: \n", cov_df
    return cov_df


def get_vifs(df):
    X = sm.add_constant(df)
    col_num = X.shape[1]
    df = X.ix[:, 1:]
    vif_list = [variance_inflation_factor(np.array(X), i) for i in np.arange(1, col_num, 1)]
    result = Series(vif_list, df.columns)
    print "VIF of all columns are: \n", result
    return result

