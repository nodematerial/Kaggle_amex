import pandas as pd
import numpy as np


def ensemble(sample, name, predictions, weights, method='avg'):
    if method == 'avg':
        weights_sum = 0
        predictions_sum = np.zeros_like(predictions[0])
        for prediction, weight in zip(predictions, weights):
            weights_sum += weight
            predictions_sum += prediction * weight
        res = predictions_sum / weights_sum
    elif method == 'min':
        res = np.minimum(*predictions)
    elif method == 'max':
        res = np.maximum(*predictions)
    sample['prediction'] = res
    sample.to_csv(f'{name}.csv', index=False)
    return None


sample = pd.read_csv('sample_submission.csv')
a = pd.read_csv('gbdt1.csv')['prediction']  # gbdt1
b = pd.read_csv('gbdt2.csv')['prediction']  # gbdt2
c = pd.read_csv('o_07987.csv')['prediction']  # cv07987
d = pd.read_csv('dart2.csv')['prediction']

ensemble(sample, 'ensemble_acd_111_min', [a, c, d], [1, 1, 1], method='min')
