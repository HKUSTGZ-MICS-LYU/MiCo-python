import csv 
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score


class WeightedEnsemble:
    def __init__(self, weight_lin=0.5):
        self.lin = LinearRegression()
        self.xgb = XGBRegressor()
        self.wlin = weight_lin
        self.wxgb = 1.0 - weight_lin
        assert self.wlin + self.wxgb == 1.0, "Weights must sum to 1.0"
    def fit(self, X, y):
        self.lin.fit(X, y)
        self.xgb.fit(X, y)
    def predict(self, X):
        y_lin = self.lin.predict(X)
        y_xgb = self.xgb.predict(X)
        y = self.wlin * y_lin + self.wxgb * y_xgb
        return y

class ResidualEnsemble:
    def __init__(self):
        self.lin = XGBRegressor(booster='gblinear', random_state=42)
        self.xgb = XGBRegressor(random_state=42)
    def fit(self, X, y):
        self.lin.fit(X, y)
        # Get residuals
        y_lin = self.lin.predict(X)
        residuals = y - y_lin
        self.xgb.fit(X, residuals)
    def predict(self, X):
        y_lin = self.lin.predict(X)
        y_xgb = self.xgb.predict(X)
        y = y_lin + y_xgb
        return y

def get_mico_matmul_proxy(mico_type: str = 'small'):
    # Load Dataset
    with open(f'benchmark_results/mico_{mico_type}_bitlinear_test.csv', 'r') as f:
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
    Q_MAX = np.max([QA, QW], axis=0)
    BMACS = MACS * Q_MAX
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    # X = np.column_stack((N, M, K, BMACS, W_LOADS, A_LOADS, QW, QA))
    # X = np.column_stack((N, M, K, QW, QA))
    # reg = XGBRegressor()

    y = latency

    X = np.column_stack((BMACS, W_LOADS, A_LOADS))
    if mico_type == 'high':
        reg = WeightedEnsemble(1.0)
    elif mico_type == "small":
        reg = WeightedEnsemble(0.8)
    else:
        reg = WeightedEnsemble(0.8)
    reg.fit(X, y)
    return reg

def get_mico_conv2d_proxy(mico_type: str = 'small'):
    # Load Dataset
    with open(f'benchmark_results/mico_{mico_type}_bitconv2d_test.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)
    
    H,W,C,K,Ks = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    QA = data[:, 5]
    QW = data[:, 6]

    latency = data[:, -1]

    H_out = (H - Ks) + 1
    W_out = (W - Ks) + 1

    MACS = H_out * W_out * C * K * Ks * Ks
    Q_MAX = np.max([QA, QW], axis=0)
    BMACS = MACS * Q_MAX
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    # X = np.column_stack((C, W, K, W_out, Ks, BMACS, W_LOADS, A_LOADS, QW, QA))
    # X = np.column_stack((H, W, C, K, W_out, H_out, Ks, QW, QA))

    y = latency

    X = np.column_stack((BMACS, W_LOADS, A_LOADS))
    if mico_type == 'high':
        reg = WeightedEnsemble(0.6)
    elif mico_type == "small":
        reg = WeightedEnsemble(0.1)
    else:
        reg = WeightedEnsemble(0.1)
    reg.fit(X, y)
    return reg

def get_mico_misc_kernel_proxy(mico_type: str, kernel_type: str, kernel_args: list):
    # Load Dataset
    with open(f'benchmark_results/mico_{mico_type}_{kernel_type}_test.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)
    x = data[:, :-1]
    y = data[:, -1]

    reg = XGBRegressor()
    reg.fit(x, y)

    pred = reg.predict(kernel_args)
    return pred


def get_bitfusion_matmul_proxy(cbops_only: bool = False, linear: bool = False):
    # Load Dataset
    with open('benchmark_results/bitfusion_matmul.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)

    M = data[:, 1]
    K = data[:, 2]
    # N = np.ones_like(M)

    QA = data[:, 3]
    QW = data[:, 4]

    latency = data[:, -1]

    MACS = M * K
    BMACS = MACS * np.max([QA, QW], axis=0)
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    y = latency
    # CBOPs Features
    X_cbops = np.column_stack((BMACS, W_LOADS, A_LOADS)) 

    # Detailed Features
    X_cbops_plus = np.column_stack((BMACS, W_LOADS, A_LOADS, M, K))

    X = X_cbops if cbops_only else X_cbops_plus

    # model = LinearRegression() if linear else XGBRegressor(booster='gblinear')
    model = LinearRegression() if linear else ResidualEnsemble()

    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Bitfusion MatMul Proxy Scores:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAPE: {mape*100:.2f}%, R2: {r2:.4f}")

    # Return Proxy Trained on All Data
    model.fit(X, y)
    return model

def get_bitfusion_conv2d_proxy(cbops_only: bool = False, linear: bool = False):
    # Load Dataset
    with open('benchmark_results/bitfusion_conv2d.csv', 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)

    H,W,C,K,Ks = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    QA = data[:, 5]
    QW = data[:, 6]

    latency = data[:, -1]

    H_out = (H - Ks) + 1
    W_out = (W - Ks) + 1
    MAC = H_out * W_out * C * K * Ks * Ks
    Q_MAX = np.max([QA, QW], axis=0)
    BMACS = MAC * Q_MAX
    W_LOADS = QW * MAC
    A_LOADS = QA * MAC

    y = latency
    # CBOPs Features
    X_cbops = np.column_stack((BMACS, W_LOADS, A_LOADS)) 

    # Detailed Features
    X_cbops_plus = np.column_stack((BMACS, W_LOADS, A_LOADS, H, W, C, K, Ks))

    X = X_cbops if cbops_only else X_cbops_plus

    # model = LinearRegression() if linear else XGBRegressor(booster='gblinear')
    model = LinearRegression() if linear else ResidualEnsemble()

    # Evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Bitfusion Conv2D Proxy Scores:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MAPE: {mape*100:.2f}%, R2: {r2:.4f}")

    # Return Proxy Trained on All Data
    model.fit(X, y)
    return model

if __name__ == "__main__":
    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()