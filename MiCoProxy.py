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

class LutProxy:
    # For Validation Only - Not a real proxy predictor
    def __init__(self, **kwargs):
        self.lut = {}
    def nearest_neighbor(self, x):
        # print(f"Warning: Extrapolating for unseen input {x} using nearest neighbor in LUT.")
        # Find the closest key in the LUT to x
        closest_key = min(self.lut.keys(), key=lambda k: np.linalg.norm(np.array(k) - np.array(x)))
        return self.lut[closest_key]
    def fit(self, X, y):
        for x, latency in zip(X, y):
            self.lut[tuple(x)] = latency
    def predict(self, X):
        X = np.array(X, dtype=int)
        return np.array([self.lut.get(tuple(x), self.nearest_neighbor(x)) for x in X])

class DirectSum:
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.sum(X, axis=1)

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

class MiCoProxy:
    def __init__(self, model, preprocess = 'raw', train_ratio = 1.0, seed = 42, 
                 train_x=None, train_y=None):
        self.model = model
        self.seed = seed
        self.train_ratio = train_ratio
        self.train_x = train_x
        self.train_y = train_y
        self.preprocess_dict = {
            "macs+": self._get_mac_plus_features,
            "raw": self._get_raw_features,
            "raw+": self._get_raw_plus_features,
            "bops": self._get_bops_features,
            "bops+": self._get_bops_plus_features,
            "cbops": self._get_cbops_features,
            "cbops+": self._get_cbops_plus_features
        }
        self.preprocess = self.preprocess_dict[preprocess]
    def _get_mac_plus_features(self, X):
        MACS = X[:, 0]
        QA = X[:, -2]
        QW = X[:, -1]
        return np.column_stack((MACS, QA, QW))
    def _get_raw_features(self, X):
        return X
    def _get_bops_features(self, X):
        MACS = X[:, 0]
        QA = X[:, -2]
        QW = X[:, -1]
        BOPS = MACS * QW * QA
        return np.column_stack((BOPS,))
    def _get_cbops_features(self, X):
        MACS = X[:, 0]
        QA = X[:, -2]
        QW = X[:, -1]
        Q_MAX = np.max(X[:, -2:], axis=1)
        BMACS = MACS * Q_MAX
        W_LOADS = MACS * QW
        A_LOADS = MACS * QA
        return np.column_stack((BMACS, W_LOADS, A_LOADS))
    
    def _get_raw_plus_features(self, X):
        QA = X[:, -2]
        QW = X[:, -1]
        Q_MIX = np.array(QA != QW, dtype=int)
        PACK_W = np.floor(32.0 / QW)
        PACK_A = np.floor(32.0 / QA)
        return np.column_stack((X, Q_MIX, PACK_W, PACK_A))
    
    def _get_cbops_plus_features(self, X):
        cbops = self._get_cbops_features(X)
        return np.column_stack((cbops, X))
    
    def _get_bops_plus_features(self, X):
        bops = self._get_bops_features(X)
        return np.column_stack((bops, X))

    def set_train_ratio(self, train_ratio: float):
        """Set training data ratio for experiments."""
        self.train_ratio = train_ratio
        
    def fit(self, X=None, y=None, train_ratio=None):
        # Apply train_ratio if less than 1.0
        if X is None or y is None:
            X = self.train_x
            y = self.train_y
        if train_ratio is not None:
            self.train_ratio = train_ratio
        if self.train_ratio < 1.0:
            total = len(X)
            subset_size = int(total * self.train_ratio)
            np.random.seed(self.seed)
            indices = np.random.choice(total, subset_size, replace=False)
            X = X[indices]
            y = y[indices]
        X = self.preprocess(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.preprocess(X)
        return self.model.predict(X)


class TwoStageProxy(MiCoProxy):
    """
    Two-stage proxy predictor that separates base latency (INT8) from speedup prediction.
    
    Inherits feature extraction methods and train_ratio handling from MiCoProxy.
    
    Stage 1: Base Latency Predictor - trained on INT8 data without QA, QW, CBOPs features
    Stage 2: Speedup Predictor - trained on non-INT8 data with precision features
    
    Final prediction = base_latency * speedup
    """
    def __init__(self, base_model, speedup_model, base_preprocess='raw', 
                 speedup_preprocess='cbops+', train_ratio=1.0, seed=42,
                 train_x=None, train_y=None):
        # Initialize parent class with dummy model (not used directly)
        super().__init__(model=None, preprocess='raw', train_ratio=train_ratio, 
                        seed=seed, train_x=train_x, train_y=train_y)
        
        self.base_model = base_model
        self.speedup_model = speedup_model
        
        # Base predictor uses features without precision info
        self.base_preprocess_dict = {
            "raw": self._get_base_features,
            "macs_only": self._get_macs_only_features,
        }
        self.base_preprocess = self.base_preprocess_dict.get(base_preprocess, self._get_base_features)
        
        # Speedup predictor uses precision-aware features (reuse parent methods)
        self.speedup_preprocess = self.preprocess_dict[speedup_preprocess]
    
    def _get_base_features(self, X):
        """Extract features without QA, QW for base latency prediction."""
        # X columns: MACS, M, K, ..., QA, QW
        # Return all except last 2 columns (QA, QW)
        return X[:, :-2]
    
    def _get_macs_only_features(self, X):
        """Use only MACS for base prediction."""
        return X[:, [0]]  # MACS is first column
    
    def fit(self, X=None, y=None, train_ratio=None):
        """
        Fit both base and speedup models.
        
        Stage 1: Train base model on INT8 data (QA=8, QW=8)
        Stage 2: Train speedup model on non-INT8 data
        """
        # Use parent's logic for handling X, y, and train_ratio
        if X is None or y is None:
            X = self.train_x
            y = self.train_y
        
        if train_ratio is not None:
            self.train_ratio = train_ratio
        

        # Separate INT8 and non-INT8 data
        # Last two columns are QA and QW
        int8_mask = (X[:, -2] == 8) & (X[:, -1] == 8)
        
        X_int8 = X[int8_mask]
        y_int8 = y[int8_mask]
        
        X_other = X[~int8_mask]
        y_other = y[~int8_mask]
        def split_train_ratio(X, y, train_ratio):
            if train_ratio < 1.0:
                total = len(X)
                subset_size = int(total * train_ratio)
                np.random.seed(self.seed)
                indices = np.random.choice(total, subset_size, replace=False)
                return X[indices], y[indices]
            return X, y
        
        X_int8, y_int8 = split_train_ratio(X_int8, y_int8, self.train_ratio)
        X_other, y_other = split_train_ratio(X_other, y_other, self.train_ratio)
        
        # Stage 1: Train base model on INT8 data
        X_base = self.base_preprocess(X_int8)
        self.base_model.fit(X_base, y_int8)
        
        # Stage 2: Train speedup model on non-INT8 data
        # Compute speedup as ratio: y_other / base_prediction(X_other)
        X_base_other = self.base_preprocess(X_other)
        base_pred_other = self.base_model.predict(X_base_other)
        
        # Compute speedup (actually slowdown since lower precision is often faster)
        # To avoid division by zero, add small epsilon
        speedup = y_other / (base_pred_other + 1e-6)
        
        X_speedup = self.speedup_preprocess(X_other)
        self.speedup_model.fit(X_speedup, speedup)
    
    def predict(self, X):
        """
        Predict latency using two-stage approach.
        
        Returns: base_latency * speedup_factor
        """
        # Get base latency prediction
        X_base = self.base_preprocess(X)
        base_latency = self.base_model.predict(X_base)
        
        # Get speedup prediction
        X_speedup = self.speedup_preprocess(X)
        speedup = self.speedup_model.predict(X_speedup)
        
        # Final prediction
        return base_latency * speedup


REGRESSOR_FACTORIES = {
    'LogRandomForest': lambda: LogRandomForestRegressor(random_state=42),
    'LogXGBRegressor': lambda: LogXGBRegressor(random_state=42),
    'XGBRegressor': lambda: XGBRegressor(random_state=42),
    'RandomForest': lambda: RandomForestRegressor(random_state=42),
}

def get_proxy(profile_dataset: str, kernel_type: str = 'matmul',
              preprocess: str = None, regressor: str = None):
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
    if regressor:
        model_factories = {regressor: REGRESSOR_FACTORIES[regressor]}
    else:
        model_factories = {
            'LogRandomForest': REGRESSOR_FACTORIES['LogRandomForest'],
        }

    if preprocess:
        feature_sets = [preprocess]
    else:
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

    # Create a fresh instance of best model with train_ratio config
    best_model = MiCoProxy(best_model_factory(), preprocess=best_features_name, 
                           train_x=X, train_y=y)
    best_model.fit()
    return best_model


def get_two_stage_proxy(profile_dataset: str, kernel_type: str = 'matmul'):
    """
    Create and train a two-stage proxy predictor with cross-validation.
    
    Stage 1: Base latency predictor trained on INT8 data
    Stage 2: Speedup predictor trained on non-INT8 data
    """
    # Load Dataset
    with open(profile_dataset, 'r') as f:
        csv_data = csv.reader(f)
        data = []
        next(csv_data)  # skip header
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
            MACS = MACS / 16  # N is always 16 in Bitfusion matmul profiles
        # N is not used for features
        RAW = (MACS, M, K, QA, QW)
    elif kernel_type == 'conv2d':
        H, W, C, K, Ks, S = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]
        QA = data[:, 6]
        QW = data[:, 7]
        latency = data[:, -1]
        H_out = (H - Ks) / S + 1
        W_out = (W - Ks) / S + 1
        MACS = H_out * W_out * C * K * Ks * Ks
        RAW = (MACS, H, W, C, K, Ks, S, QA, QW)
    
    y = latency
    X = np.column_stack(RAW)
    
    # Model factories for base and speedup models
    base_model_factories = {
        # 'LogRandomForest': lambda: LogRandomForestRegressor(random_state=42),
        'LogXGBRegressor': lambda: LogXGBRegressor(random_state=42)
    }
    
    speedup_model_factories = {
        # 'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'XGBRegressor': lambda: XGBRegressor(random_state=42),
        # 'LogRandomForest': lambda: LogRandomForestRegressor(random_state=42),
        # 'LogXGBsRegressor': lambda: LogXGBRegressor(random_state=42)
    }
    
    # Feature sets for base (without precision) and speedup (with precision)
    base_feature_sets = ['raw'] # raw, macs_only
    speedup_feature_sets = ['raw'] # raw, bops, bops+, cbops, cbops+
    
    best_mape = float('inf')
    best_base_factory = None
    best_speedup_factory = None
    best_base_name = None
    best_speedup_name = None
    best_base_features = None
    best_speedup_features = None
    
    print(f"\nTwo-Stage MiCo {kernel_type} Proxy - Cross-Validation Results:")
    print("=" * 80)
    
    # Try all combinations
    for base_features in base_feature_sets:
        for speedup_features in speedup_feature_sets:
            for base_name, base_factory in base_model_factories.items():
                for speedup_name, speedup_factory in speedup_model_factories.items():
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    mapes = []
                    r2s = []
                    maes = []
                    
                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        
                        # Create two-stage proxy
                        model = TwoStageProxy(
                            base_factory(), speedup_factory(),
                            base_preprocess=base_features,
                            speedup_preprocess=speedup_features
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mapes.append(mean_absolute_percentage_error(y_test, y_pred))
                        r2s.append(r2_score(y_test, y_pred))
                        maes.append(mean_absolute_error(y_test, y_pred))
                    
                    mean_mape = np.mean(mapes)
                    mean_r2 = np.mean(r2s)
                    mean_mae = np.mean(maes)
                    
                    config = f"[{base_features:10s}+{speedup_features:7s}] {base_name:20s} + {speedup_name:20s}"
                    print(f"  {config}: MAPE={mean_mape*100:6.2f}%, R2={mean_r2:7.4f}, MAE={mean_mae:.2f}")
                    
                    # Track best configuration
                    if mean_mape < best_mape:
                        best_mape = mean_mape
                        best_base_factory = base_factory
                        best_speedup_factory = speedup_factory
                        best_base_name = base_name
                        best_speedup_name = speedup_name
                        best_base_features = base_features
                        best_speedup_features = speedup_features
    
    print("=" * 80)
    print(f"Best Two-Stage Model: {best_base_name} + {best_speedup_name}")
    print(f"Base features: {best_base_features}, Speedup features: {best_speedup_features}")
    print(f"Best Cross-Validation MAPE: {best_mape*100:6.2f}%")
    
    # Create and train final model with best configuration
    best_model = TwoStageProxy(
        best_base_factory(), best_speedup_factory(),
        base_preprocess=best_base_features,
        speedup_preprocess=best_speedup_features,
        train_x=X, train_y=y
    )
    best_model.fit()
    return best_model


def get_mico_matmul_proxy(mico_type: str = 'small', two_stage=False,
                         preprocess=None, regressor=None):
    if two_stage:
        return get_two_stage_proxy(
            f'benchmark_results/mico_{mico_type}_matmul_zoo.csv',
            'matmul'
        )
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_matmul_zoo.csv',
        'matmul', preprocess=preprocess, regressor=regressor
    )

def get_mico_conv2d_proxy(mico_type: str = 'small', two_stage=False,
                          preprocess=None, regressor=None):
    if two_stage:
      return get_two_stage_proxy(
        f'benchmark_results/mico_{mico_type}_conv2d_zoo.csv',
        'conv2d'
      )  
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_conv2d_zoo.csv',
        'conv2d', preprocess=preprocess, regressor=regressor
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

def get_bitfusion_matmul_proxy(two_stage=False, preprocess=None, regressor=None):
    if two_stage:
        return get_two_stage_proxy('benchmark_results/bitfusion_matmul_zoo.csv', 'matmul')
    return get_proxy('benchmark_results/bitfusion_matmul_zoo.csv', 'matmul',
                     preprocess=preprocess, regressor=regressor)

def get_bitfusion_conv2d_proxy(two_stage=False, preprocess=None, regressor=None):
    if two_stage:
        return get_two_stage_proxy('benchmark_results/bitfusion_conv2d_zoo.csv', 'conv2d')
    return get_proxy('benchmark_results/bitfusion_conv2d_zoo.csv', 'conv2d',
                     preprocess=preprocess, regressor=regressor)

def get_host_matmul_proxy(opt="opt", two_stage=False):
    if two_stage:
        return get_two_stage_proxy(
            f'benchmark_results/host_{opt}_matmul_zoo.csv',
            'matmul'
        )
    return get_proxy(
        f'benchmark_results/host_{opt}_matmul_zoo.csv',
        'matmul'
    )

def get_host_conv2d_proxy(opt="opt", two_stage=False):
    if two_stage:
        return get_two_stage_proxy(
            f'benchmark_results/host_{opt}_conv2d_zoo.csv',
            'conv2d'
        )
    return get_proxy(
        f'benchmark_results/host_{opt}_conv2d_zoo.csv',
        'conv2d'
    )

if __name__ == "__main__":
    # Test Bitfusion proxies with cross-validation
    # print("\n" + "="*80)
    print("BITFUSION PROXY TUNING")
    print("="*80)
    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()
    
    # Test MiCo proxies with cross-validation
    print("\n" + "="*80)
    print("MICO PROXY TUNING")
    print("="*80)
    
    # # Test for 'small' mico type
    print("\n### Testing MiCo 'small' type ###")
    mico_small_matmul_proxy = get_mico_matmul_proxy(mico_type='small')
    mico_small_conv2d_proxy = get_mico_conv2d_proxy(mico_type='small')
    
    # # # Test for 'high' mico type
    print("\n### Testing MiCo 'high' type ###")
    mico_high_matmul_proxy = get_mico_matmul_proxy(mico_type='high')
    mico_high_conv2d_proxy = get_mico_conv2d_proxy(mico_type='high')
    
    # # Test Host MatMul proxy with cross-validation
    # print("\n" + "="*80)
    # print("HOST PROXY TUNING")
    # print("="*80)

    # print("\n### Testing Host 'opt' proxy ###")
    # host_matmul_proxy = get_host_matmul_proxy(opt="opt")
    # host_conv2d_proxy = get_host_conv2d_proxy(opt="opt")

    # print("\n### Testing Host 'lut' proxy ###")
    # host_matmul_proxy_lut = get_host_matmul_proxy(opt="lut")

    # print("\n" + "="*80)
    # print("ALL PROXY TUNING COMPLETED")
    # print("="*80)