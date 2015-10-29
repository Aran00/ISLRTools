__author__ = 'ryu'

from pandas import DataFrame, Series
import numpy as np


def prob_to_value(x, threshold):
    return 1 if x > threshold else 0


def get_value(arr, i):
    '''
    Can be Series, ndarray(1-d) or ordiary list
    For Series, arr[i] would return an element that uses i as index
    '''
    if i < 0 or i >= len(arr):
        return None
    if isinstance(arr, Series):
        # Use iloc here
        return arr.iloc[i]
    else:
        return arr[i]


def get_idx(value, zero_one_col_texts):
    if isinstance(value, basestring):
        return 0 if value == zero_one_col_texts[0] else 1
    else:
        return int(value)


def output_table_with_prob(predict_probs, real_values, threshold=0.5, zero_one_col_texts=[0, 1]):
    output_data = DataFrame([[0, 0], [0, 0]], columns=zero_one_col_texts, index=zero_one_col_texts)
    if isinstance(predict_probs, Series):
        pred = predict_probs.map(lambda x: 1 if x > threshold else 0)
    else:
        vfunc = np.vectorize(prob_to_value)
        pred = vfunc(predict_probs, threshold)
    length = len(pred)
    for i in xrange(0, length):  # or len(obj)
        output_data.ix[get_idx(get_value(pred, i), zero_one_col_texts),
                       get_idx(get_value(real_values, i), zero_one_col_texts)] += 1
    print output_data
    print "The predict correctness rate is", (output_data.ix[0, 0] + output_data.ix[1, 1])/float(length)
    print "The correctness of 0-0 is", float(output_data.ix[0, 0])/(output_data.ix[0, 0] + output_data.ix[0, 1])
    print "The correctness of 1-1 is", float(output_data.ix[1, 1])/(output_data.ix[1, 0] + output_data.ix[1, 1])


''' params: nd-array '''
def output_table(pred_values, real_values):
    ''' Just compare 2 lists and output the compare result '''
    length = len(pred_values)
    assert(length == len(real_values))
    unique_values = np.unique(np.hstack((pred_values, real_values)))
    output_data = DataFrame([[0, 0], [0, 0]], columns=unique_values, index=unique_values)
    for i in xrange(len(pred_values)):
        output_data.ix[pred_values[i], real_values[i]] += 1
    print output_data
    pred_right_list = [output_data.ix[unique_values[i], unique_values[i]] for i in xrange(len(unique_values))]
    print "The predict correctness rate is", np.sum(pred_right_list)/float(length)