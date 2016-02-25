__author__ = 'ryu'

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets.base import Bunch


def load_data(dataset_name, y_col, x_cols=None, drop_na=True, transform_label=True, **args):
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.join(_ROOT, 'data', '%s.csv' % dataset_name)
    data = pd.read_csv(file_name, **args)
    if drop_na:
        data = data.dropna()
    if transform_label:
        data = _transform_label(data)
    X, y = _prepare_data(data, y_col, x_cols)
    return Bunch(data=X, target=y, full=data, feature_names=x_cols, target_property=y_col)


def _prepare_data(data, y_col, x_cols=None):
    y = data[y_col]
    if x_cols is None:
        x_cols = data.columns.tolist()
        x_cols.remove(y_col)
    X = data[x_cols]
    return X, y


def _transform_label(data):
    for col in data.columns.tolist():
        if isinstance(data.iloc[0][col], str):
            le = preprocessing.LabelEncoder()
            le.fit(np.unique(data[col]))
            data[col] = le.transform(data[col].values)
    return data