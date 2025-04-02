import csv 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def get_mico_proxy(mico_type: str = 'small', model_type: str = 'linear'):
    # Load Dataset
    with open(f'benchmark_results/mico_{mico_type}.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)
    
    N = data[:, 0]
    M = data[:, 1]
    K = data[:, 2]

    QA = data[:, 3]
    QW = data[:, 4]

    latency = data[:, -1]

    MACS = N * M * K
    BMACS = MACS * np.max([QA, QW], axis=0)
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    X = np.column_stack((BMACS, W_LOADS, A_LOADS))
    y = latency

    if model_type == 'linear':
        reg = LinearRegression()
    else:
        raise NotImplementedError
    reg.fit(X, y)
    return reg

def get_bitfusion_proxy():
    # Load Dataset
    with open('benchmark_results/bitfusion.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)

    N = data[:, 0]
    M = data[:, 1]
    K = data[:, 2]

    QA = data[:, 3]
    QW = data[:, 4]

    latency = data[:, -1]

    MACS = N * M * K
    BMACS = MACS * np.max([QA, QW], axis=0)
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    X = np.column_stack((BMACS, W_LOADS, A_LOADS))
    y = latency

    reg = LinearRegression()
    reg.fit(X, y)
    return reg