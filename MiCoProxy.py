import csv 
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score


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

class ResidualEnsemble:
    def __init__(self, random_state=42):
        self.lin = XGBRegressor(booster='gblinear', random_state=random_state)
        self.xgb = XGBRegressor(random_state=random_state)
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

class LogXGBRegressor:
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)
    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
    def predict(self, X):
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

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

    y = latency

    # Feature sets to try
    X_cbops = np.column_stack((BMACS, W_LOADS, A_LOADS))
    X_cbops_plus = np.column_stack((BMACS, W_LOADS, A_LOADS, N, M, K))
    X_raw = np.column_stack((N, M, K, QW, QA))

    # Model factories - functions that create new model instances
    model_factories = {
        'LinearRegression': lambda: LinearRegression(),
        'Ridge': lambda: Ridge(),
        'Lasso': lambda: Lasso(),
        'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'XGBRegressor': lambda: XGBRegressor(random_state=42),
        'XGBRegressor_gblinear': lambda: XGBRegressor(booster='gblinear', random_state=42),
        'WeightedEnsemble_0.5': lambda: WeightedEnsemble(0.5),
        'WeightedEnsemble_0.8': lambda: WeightedEnsemble(0.8),
        'WeightedEnsemble_1.0': lambda: WeightedEnsemble(1.0),
        'ResidualEnsemble': lambda: ResidualEnsemble(random_state=42),
        'LogXGBRegressor': lambda: LogXGBRegressor(random_state=42),
    }

    feature_sets = {
        'CBOPs': X_cbops,
        'CBOPs+': X_cbops_plus,
        'Raw': X_raw,
    }

    best_mape = float('inf')
    best_model_factory = None
    best_model_name = None
    best_features_name = None
    best_X = None

    print(f"\nMiCo {mico_type.upper()} MatMul Proxy - Cross-Validation Results:")
    print("=" * 80)

    # Try all combinations
    for feature_name, X in feature_sets.items():
        for model_name, model_factory in model_factories.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mapes = []
            r2s = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Create a fresh model instance for each fold
                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mapes.append(mean_absolute_percentage_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))
            
            mean_mape = np.mean(mapes)
            mean_r2 = np.mean(r2s)
            
            print(f"  [{feature_name:8s}] {model_name:25s}: MAPE={mean_mape*100:6.2f}%, R2={mean_r2:7.4f}")
            
            # Track best model factory
            if mean_mape < best_mape:
                best_mape = mean_mape
                best_model_factory = model_factory
                best_model_name = model_name
                best_features_name = feature_name
                best_X = X

    print("=" * 80)
    print(f"Best Model: {best_model_name} with {best_features_name} features")
    print(f"Best MAPE: {best_mape*100:.2f}%, R2 score during CV")
    print()

    # Create a fresh instance of best model and train on all data
    best_model = best_model_factory()
    best_model.fit(best_X, y)
    return best_model

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
    Q_MUL = QA * QW
    BOPS = Q_MUL * MACS
    BMACS = MACS * Q_MAX
    W_LOADS = QW * MACS
    A_LOADS = QA * MACS

    y = latency

    # Feature sets to try
    X_cbops = np.column_stack((BMACS, W_LOADS, A_LOADS))
    X_cbops_plus = np.column_stack((BMACS, W_LOADS, A_LOADS, H, W, C, K, Ks))
    X_bops_plus = np.column_stack((BOPS, H, W, C, K, Ks))
    X_raw = np.column_stack((H, W, C, K, Ks, QW, QA))

    # Model factories - functions that create new model instances
    model_factories = {
        'LinearRegression': lambda: LinearRegression(),
        'Ridge': lambda: Ridge(),
        'Lasso': lambda: Lasso(),
        'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'XGBRegressor': lambda: XGBRegressor(random_state=42),
        'XGBRegressor_gblinear': lambda: XGBRegressor(booster='gblinear', random_state=42),
        'WeightedEnsemble_0.1': lambda: WeightedEnsemble(0.1),
        'WeightedEnsemble_0.5': lambda: WeightedEnsemble(0.5),
        'WeightedEnsemble_0.6': lambda: WeightedEnsemble(0.6),
        'ResidualEnsemble': lambda: ResidualEnsemble(random_state=42),
        'LogXGBRegressor': lambda: LogXGBRegressor(random_state=42),
    }

    feature_sets = {
        'CBOPs': X_cbops,
        'CBOPs+': X_cbops_plus,
        'BOPs+': X_bops_plus,
        'Raw': X_raw,
    }

    best_mape = float('inf')
    best_model_factory = None
    best_model_name = None
    best_features_name = None
    best_X = None

    print(f"\nMiCo {mico_type.upper()} Conv2D Proxy - Cross-Validation Results:")
    print("=" * 80)

    # Try all combinations
    for feature_name, X in feature_sets.items():
        for model_name, model_factory in model_factories.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mapes = []
            r2s = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Create a fresh model instance for each fold
                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mapes.append(mean_absolute_percentage_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))
            
            mean_mape = np.mean(mapes)
            mean_r2 = np.mean(r2s)
            
            print(f"  [{feature_name:8s}] {model_name:25s}: MAPE={mean_mape*100:6.2f}%, R2={mean_r2:7.4f}")
            
            # Track best model factory
            if mean_mape < best_mape:
                best_mape = mean_mape
                best_model_factory = model_factory
                best_model_name = model_name
                best_features_name = feature_name
                best_X = X

    print("=" * 80)
    print(f"Best Model: {best_model_name} with {best_features_name} features")
    print(f"Best MAPE: {best_mape*100:.2f}%, R2 score during CV")
    print()

    # Create a fresh instance of best model and train on all data
    best_model = best_model_factory()
    best_model.fit(best_X, y)
    return best_model

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
    model = LinearRegression() if linear else ResidualEnsemble(random_state=42)

    # Evaluate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mapes = []
    r2s = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create a new instance for each fold to avoid data leakage if the model has state
        # For simple classes we can just re-fit, but let's be safe if we can.
        # Since we don't have a clone method, we rely on fit() resetting state or overwriting it.
        # XGBRegressor and LinearRegression fit() overwrites.
        # ResidualEnsemble and LogXGBRegressor fit() also overwrites.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mapes.append(mean_absolute_percentage_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))
    
    print("Bitfusion MatMul Proxy Scores (5-Fold CV):")
    print(f"  MAPE: {np.mean(mapes)*100:.2f}%, R2: {np.mean(r2s):.4f}")

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
    Q_MUL = QA * QW
    BOPS = Q_MUL * MAC
    BMACS = MAC * Q_MAX
    W_LOADS = QW * MAC
    A_LOADS = QA * MAC

    y = latency
    # CBOPs Features
    X_cbops = np.column_stack((BMACS, W_LOADS, A_LOADS)) 

    # CBOPs + Layer Features
    X_cbops_plus = np.column_stack((BMACS, W_LOADS, A_LOADS, H, W, C, K, Ks))

    # BOPs Features
    X_bops_plus = np.column_stack((BOPS, H, W, C, K, Ks))

    # RAW Features
    # X_cbops_plus = np.column_stack((QW, QA, H, W, C, K, Ks))

    X = X_cbops if cbops_only else X_cbops_plus

    # model = LinearRegression() if linear else XGBRegressor(booster='gblinear')
    model = LinearRegression() if linear else LogXGBRegressor(random_state=42)
    # Evaluate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mapes = []
    r2s = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mapes.append(mean_absolute_percentage_error(y_test, y_pred))
        r2s.append(r2_score(y_test, y_pred))

    print("Bitfusion Conv2D Proxy Scores (5-Fold CV):")
    print(f"  MAPE: {np.mean(mapes)*100:.2f}%, R2: {np.mean(r2s):.4f}")

    # Return Proxy Trained on All Data
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Test Bitfusion proxies with cross-validation
    print("\n" + "="*80)
    print("BITFUSION PROXY TUNING")
    print("="*80)
    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()
    
    # Test MiCo proxies with cross-validation
    print("\n" + "="*80)
    print("MICO PROXY TUNING")
    print("="*80)
    
    # Test for 'small' mico type
    print("\n### Testing MiCo 'small' type ###")
    mico_small_matmul_proxy = get_mico_matmul_proxy(mico_type='small')
    mico_small_conv2d_proxy = get_mico_conv2d_proxy(mico_type='small')
    
    # Test for 'high' mico type
    print("\n### Testing MiCo 'high' type ###")
    mico_high_matmul_proxy = get_mico_matmul_proxy(mico_type='high')
    mico_high_conv2d_proxy = get_mico_conv2d_proxy(mico_type='high')
    
    print("\n" + "="*80)
    print("ALL PROXY TUNING COMPLETED")
    print("="*80)