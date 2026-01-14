import csv 
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_percentage_error, 
    r2_score, 
    mean_absolute_error
)


class WeightedEnsemble:
    def __init__(self, weight_lin=0.5):
        self.lin = LinearRegression()
        self.xgb = XGBRegressor(random_state=42)
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

class LogXGBRegressor:
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)
    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
    def predict(self, X):
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

class LogRandomForestRegressor:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
    def predict(self, X):
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

class ResidualEnsemble:
    def __init__(self, **kwargs):
        self.lin = XGBRegressor(booster='gblinear', **kwargs)
        self.xgb = XGBRegressor(**kwargs)
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
    
class LogResidualEnsemble:
    def __init__(self, **kwargs):
        self.lin = LogXGBRegressor(booster='gblinear', **kwargs)
        self.xgb = XGBRegressor(**kwargs)
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

class MiCoProxy:
    def __init__(self, model, preprocess = 'raw'):
        self.model = model
        self.preprocess_type = preprocess

    def preprocess(self, X):
        # Linear Features (MACS, M, K, QA, QW)
        # Conv2D Features (MACS, H, W, C, K, Ks, S, QA, QW)
        MACS = X[:, 0]
        QA = X[:, -2]
        QW = X[:, -1]
        if self.preprocess_type == 'raw':
            X_processed = X
        elif self.preprocess_type == 'bops+':
            BOPS = MACS * QW * QA
            X_processed = np.column_stack((BOPS, X))
        elif self.preprocess_type == 'cbops':
            Q_MAX = np.max(X[:, -2:], axis=1)
            BMACS = MACS * Q_MAX
            W_LOADS = MACS * QW
            A_LOADS = MACS * QA
            X_processed = np.column_stack((BMACS, W_LOADS, A_LOADS))
        elif self.preprocess_type == 'cbops+':
            Q_MAX = np.max(X[:, -2:], axis=1)
            BMACS = MACS * Q_MAX
            W_LOADS = MACS * QW
            A_LOADS = MACS * QA
            X_processed = np.column_stack((BMACS, W_LOADS, A_LOADS, X))
        
        return X_processed

    def fit(self, X, y):
        X = self.preprocess(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)

def get_proxy(profile_dataset: str, kernel_type: str = 'matmul'):
    # Load Dataset
    with open(profile_dataset, 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data) # skip header
        for row in csv_data:
            data.append(list(map(int, row)))
    data = np.array(data)
    if kernel_type == 'matmul':
        N = data[:, 0]
        M = data[:, 1]
        K = data[:, 2]

        QA = data[:, 3]
        QW = data[:, 4]

        latency = data[:, -1]
        MACS = N * M * K
        if 'bitfusion' in profile_dataset:
            MACS = MACS / 16 # N is always 16 in Bitfusion matmul profiles    
        # N is not used for features
        RAW = (MACS, M, K, QA, QW)
    elif kernel_type == 'conv2d':
        H,W,C,K,Ks,S = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]

        QA = data[:, 6]
        QW = data[:, 7]

        latency = data[:, -1]

        H_out = (H - Ks) / S + 1
        W_out = (W - Ks) / S + 1
        MACS = H_out * W_out * C * K * Ks * Ks
        RAW = (MACS, H, W, C, K, Ks, S, QA, QW)
    
    y = latency
    X = np.column_stack(RAW)

    # Model factories - functions that create new model instances
    model_factories = {
        # 'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'LogRandomForest': lambda: LogRandomForestRegressor(random_state=42),
        # 'XGBRegressor': lambda: XGBRegressor(random_state=42),
        # 'LogXGBRegressor': lambda: LogXGBRegressor(random_state=42)
    }

    # feature_sets = ['raw', 'bops+', 'cbops', 'cbops+']
    feature_sets = ['cbops+']

    best_mape = float('inf')
    best_model_factory = None
    best_model_name = None
    best_features_name = None

    print(f"\nMiCo {kernel_type} Proxy - Cross-Validation Results:")
    print("=" * 80)

    # Try all combinations
    for feature_name in feature_sets:
        for model_name, model_factory in model_factories.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mapes = []
            r2s = []
            maes = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Create a fresh model instance for each fold
                model = MiCoProxy(model_factory(), preprocess=feature_name)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mapes.append(mean_absolute_percentage_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))
                maes.append(mean_absolute_error(y_test, y_pred))
            
            mean_mape = np.mean(mapes)
            mean_r2 = np.mean(r2s)
            mean_mae = np.mean(maes)
            
            print(f"  [{feature_name:8s}] {model_name:25s}: MAPE={mean_mape*100:6.2f}%, R2={mean_r2:7.4f}, MAE={mean_mae:.2f}")
            
            # Track best model factory
            if mean_mape < best_mape:
                best_mape = mean_mape
                best_model_factory = model_factory
                best_model_name = model_name
                best_features_name = feature_name

    print("=" * 80)
    print(f"Best Model: {best_model_name} with {best_features_name} features")
    print(f"Best Cross-Validation MAPE: {best_mape*100:6.2f}%")

    # Create a fresh instance of best model and train on all data
    best_model = MiCoProxy(best_model_factory(), preprocess=best_features_name)
    best_model.fit(X, y)
    return best_model

def get_mico_matmul_proxy(mico_type: str = 'small'):
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_bitlinear_test.csv',
        'matmul'
    )

def get_mico_conv2d_proxy(mico_type: str = 'small'):
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_bitconv2d_test.csv',
        'conv2d'
    )

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

def get_bitfusion_matmul_proxy():
    return get_proxy('benchmark_results/bitfusion_matmul_samples.csv', 'matmul')

def get_bitfusion_conv2d_proxy():
    return get_proxy('benchmark_results/bitfusion_conv2d_resnet8.csv', 'conv2d')

def get_host_matmul_proxy(opt="opt"):
    return get_proxy(
        f'benchmark_results/host_{opt}_bitlinear_test.csv',
        'matmul'
    )

def get_host_conv2d_proxy(opt="opt"):
    return get_proxy(
        f'benchmark_results/host_{opt}_bitconv2d_test.csv',
        'conv2d'
    )

if __name__ == "__main__":
    # Test Bitfusion proxies with cross-validation
    print("\n" + "="*80)
    print("BITFUSION PROXY TUNING")
    print("="*80)
    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()
    
    # # Test MiCo proxies with cross-validation
    # print("\n" + "="*80)
    # print("MICO PROXY TUNING")
    # print("="*80)
    
    # # Test for 'small' mico type
    # print("\n### Testing MiCo 'small' type ###")
    # mico_small_matmul_proxy = get_mico_matmul_proxy(mico_type='small')
    # mico_small_conv2d_proxy = get_mico_conv2d_proxy(mico_type='small')
    
    # # # Test for 'high' mico type
    # print("\n### Testing MiCo 'high' type ###")
    # mico_high_matmul_proxy = get_mico_matmul_proxy(mico_type='high')
    # mico_high_conv2d_proxy = get_mico_conv2d_proxy(mico_type='high')
    
    # # Test Host MatMul proxy with cross-validation
    # print("\n" + "="*80)
    # print("HOST PROXY TUNING")
    # print("="*80)

    # # print("\n### Testing Host 'opt' proxy ###")
    # host_matmul_proxy = get_host_matmul_proxy(opt="opt")
    # host_conv2d_proxy = get_host_conv2d_proxy(opt="opt")

    # print("\n### Testing Host 'lut' proxy ###")
    # host_matmul_proxy_lut = get_host_matmul_proxy(opt="lut")

    # print("\n" + "="*80)
    # print("ALL PROXY TUNING COMPLETED")
    # print("="*80)