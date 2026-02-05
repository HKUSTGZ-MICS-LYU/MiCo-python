import csv 
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_percentage_error, 
    r2_score, 
    mean_absolute_error
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class LogMLPRegressor:
    """
    MLP-based regressor with log transformation for latency prediction.
    
    This class wraps sklearn's MLPRegressor with feature scaling and log
    transformation of target values. MLPs support true transfer learning
    via warm_start, allowing the model to continue training from pretrained
    weights when fine-tuning on new data.
    """
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=2000, 
                 learning_rate_init=0.001, early_stopping=False, 
                 validation_fraction=0.1, warm_start=True, random_state=None, **kwargs):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            warm_start=warm_start,  # Enable true transfer learning
            random_state=random_state,
            **kwargs
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X, y):
        # Scale features for better MLP training
        if not self._is_fitted:
            # First fit: learn the scaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Subsequent fits (fine-tuning): use existing scaler
            X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, np.log1p(y))
        self._is_fitted = True
    
    def finetune(self, X, y):
        """Continue training on new data (leverages warm_start)."""
        # With warm_start=True, calling fit() continues from previous weights
        return self.fit(X, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_pred_log = self.model.predict(X_scaled)
        return np.expm1(y_pred_log)


class TorchMLP(nn.Module):
    """PyTorch MLP architecture for latency prediction with transfer learning support."""
    
    def __init__(self, input_dim, hidden_dims=(128, 64, 32)):
        super(TorchMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers (feature extractor)
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final layer (head) - this will be fine-tuned or replaced for transfer learning
        self.head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features).squeeze(-1)
    
    def freeze_feature_extractor(self):
        """Freeze all layers except the head for transfer learning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True


class TorchMLPRegressor:
    """
    PyTorch-based MLP regressor with proper transfer learning support.
    
    This class implements a neural network using PyTorch with advanced
    transfer learning techniques:
    - Layer freezing: Freeze feature extractor, fine-tune only the head
    - Discriminative learning rates: Lower LR for pretrained layers
    - Gradual unfreezing: Progressively unfreeze layers during training
    
    Args:
        hidden_dims: Tuple of hidden layer dimensions (default: (128, 64, 32))
        epochs: Number of training epochs (default: 500)
        lr: Learning rate (default: 0.001)
        batch_size: Batch size for training (default: 32)
        freeze_strategy: Transfer learning strategy:
            - 'none': No freezing, fine-tune all layers
            - 'freeze_extractor': Freeze feature extractor, only train head
            - 'discriminative_lr': Use lower LR for pretrained layers
        device: Device to use ('cuda', 'cpu', or 'auto')
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, hidden_dims=(128, 64, 32), epochs=500, lr=0.001, 
                 batch_size=32, freeze_strategy='freeze_extractor',
                 device='auto', random_state=None, verbose=False):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.freeze_strategy = freeze_strategy
        self.verbose = verbose
        self.random_state = random_state
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._input_dim = None
    
    def _create_model(self, input_dim):
        """Create a new model with the given input dimension."""
        self._input_dim = input_dim
        self.model = TorchMLP(input_dim, self.hidden_dims).to(self.device)
    
    def _train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def fit(self, X, y):
        """
        Train the model from scratch.
        
        Args:
            X: Training features (numpy array)
            y: Training targets (numpy array)
        """
        # Scale features
        if not self._is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Log transform targets
        y_log = np.log1p(y)
        
        # Create model if needed
        if self.model is None or self._input_dim != X_scaled.shape[1]:
            self._create_model(X_scaled.shape[1])
        
        # Create dataloader
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_log)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        for epoch in range(self.epochs):
            loss = self._train_epoch(dataloader, optimizer, criterion)
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.6f}")
        
        self._is_fitted = True
    
    def finetune(self, X, y, epochs=None, lr=None):
        """
        Fine-tune the model on new data using the specified freeze strategy.
        
        This implements proper transfer learning by:
        1. freeze_extractor: Freezing the feature extraction layers
        2. Training only the head (final layers) on new data
        3. Optionally unfreezing for full fine-tuning
        
        Args:
            X: Fine-tuning features (numpy array)
            y: Fine-tuning targets (numpy array)
            epochs: Number of fine-tuning epochs (default: half of original epochs)
            lr: Learning rate for fine-tuning (default: 1/10 of original lr)
        """
        if not self._is_fitted:
            return self.fit(X, y)
        
        epochs = epochs or max(self.epochs // 2, 100)
        lr = lr or self.lr / 10  # Lower LR for fine-tuning
        
        # Scale features using existing scaler
        X_scaled = self.scaler.transform(X)
        y_log = np.log1p(y)
        
        # Create dataloader
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_log)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        
        if self.freeze_strategy == 'freeze_extractor':
            # Freeze feature extractor, only train head
            self.model.freeze_feature_extractor()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=lr
            )
            
            # Phase 1: Train only head
            for epoch in range(epochs // 2):
                loss = self._train_epoch(dataloader, optimizer, criterion)
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"[Frozen] Epoch {epoch+1}/{epochs//2}, Loss: {loss:.6f}")
            
            # Phase 2: Unfreeze and fine-tune all with lower LR
            self.model.unfreeze_all()
            optimizer = optim.Adam(self.model.parameters(), lr=lr / 5)
            
            for epoch in range(epochs // 2):
                loss = self._train_epoch(dataloader, optimizer, criterion)
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"[Unfrozen] Epoch {epoch+1}/{epochs//2}, Loss: {loss:.6f}")
                    
        elif self.freeze_strategy == 'discriminative_lr':
            # Use different learning rates for different layers
            optimizer = optim.Adam([
                {'params': self.model.feature_extractor.parameters(), 'lr': lr / 10},
                {'params': self.model.head.parameters(), 'lr': lr}
            ])
            
            for epoch in range(epochs):
                loss = self._train_epoch(dataloader, optimizer, criterion)
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"[Discriminative LR] Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        else:
            # No freezing - fine-tune all layers with lower LR
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                loss = self._train_epoch(dataloader, optimizer, criterion)
                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"[Full] Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Predict latency values.
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            numpy array of predicted latency values
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            y_pred_log = self.model(X_tensor).cpu().numpy()
        
        return np.expm1(y_pred_log)


class MiCoProxy:
    def __init__(self, model, preprocess = 'raw', train_ratio = 1.0, seed = 42, 
                 train_x=None, train_y=None):
        self.model = model
        self.seed = seed
        self.train_ratio = train_ratio
        self.train_x = train_x
        self.train_y = train_y
        self.preprocess_name = preprocess
        self.preprocess_dict = {
            "raw": self._get_raw_features,
            "bops": self._get_bops_features,
            "bops+": self._get_bops_plus_features,
            "cbops": self._get_cbops_features,
            "cbops+": self._get_cbops_plus_features
        }
        self.preprocess = self.preprocess_dict[preprocess]
        # Transfer learning state
        self._is_pretrained = False
        self._source_domain = None
        
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
    
    def pretrain(self, X_source, y_source):
        """
        Pretrain the model on source domain data for transfer learning.
        
        This method trains the base model on source domain data, which can
        later be fine-tuned on target domain data using the finetune() method.
        
        Args:
            X_source: Source domain features (e.g., mico_small data)
            y_source: Source domain labels (latency values)
        """
        X_source = self.preprocess(X_source)
        self.model.fit(X_source, y_source)
        self._is_pretrained = True
        self._source_x = X_source
        self._source_y = y_source
    
    def finetune(self, X_target, y_target, finetune_ratio=1.0, strategy='combined'):
        """
        Fine-tune a pretrained model on target domain data.
        
        This enables transfer learning by leveraging knowledge from the source
        domain while adapting to the target domain with limited data.
        
        Args:
            X_target: Target domain features (e.g., mico_high data)
            y_target: Target domain labels (latency values)
            finetune_ratio: Ratio of target data to use (0.0 to 1.0)
            strategy: Fine-tuning strategy:
                - 'combined': Combine source and target data for retraining
                - 'target_only': Retrain only on target data (baseline)
                - 'weighted': Weight target samples higher than source
                - 'finetune': Use model's native finetune method (for torch_mlp)
        
        Returns:
            dict: Fine-tuning results with metrics
        """
        if not self._is_pretrained and strategy != 'target_only':
            raise ValueError("Model must be pretrained before fine-tuning. "
                           "Call pretrain() first or use strategy='target_only'.")
        
        # Apply finetune_ratio to select subset of target data
        np.random.seed(self.seed)
        total = len(X_target)
        subset_size = max(1, int(total * finetune_ratio))
        indices = np.random.choice(total, subset_size, replace=False)
        X_target_subset = X_target[indices]
        y_target_subset = y_target[indices]
        
        # Preprocess target data
        X_target_processed = self.preprocess(X_target_subset)
        
        # Check if model has native finetune method (e.g., TorchMLPRegressor)
        has_finetune = hasattr(self.model, 'finetune') and callable(self.model.finetune)
        
        if strategy == 'target_only':
            # Retrain only on target data (baseline for comparison)
            self.model.fit(X_target_processed, y_target_subset)
        elif strategy == 'finetune' and has_finetune:
            # Use model's native finetune method (proper transfer learning)
            self.model.finetune(X_target_processed, y_target_subset)
        elif strategy == 'combined':
            # Combine source and target data
            X_combined = np.vstack([self._source_x, X_target_processed])
            y_combined = np.concatenate([self._source_y, y_target_subset])
            if has_finetune:
                # For torch_mlp, use finetune for better transfer
                self.model.finetune(X_combined, y_combined)
            else:
                self.model.fit(X_combined, y_combined)
        elif strategy == 'weighted':
            # Weight target samples more heavily by oversampling
            target_size = max(1, len(X_target_processed))
            weight_factor = max(1, len(self._source_x) // target_size)
            X_target_weighted = np.tile(X_target_processed, (weight_factor, 1))
            y_target_weighted = np.tile(y_target_subset, weight_factor)
            X_combined = np.vstack([self._source_x, X_target_weighted])
            y_combined = np.concatenate([self._source_y, y_target_weighted])
            if has_finetune:
                self.model.finetune(X_combined, y_combined)
            else:
                self.model.fit(X_combined, y_combined)
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
        
        return {
            'finetune_ratio': finetune_ratio,
            'target_samples_used': subset_size,
            'strategy': strategy
        }
    
    def is_pretrained(self):
        """Check if model has been pretrained for transfer learning."""
        return self._is_pretrained

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
        # 'DirectSum': lambda: DirectSum(),
        # 'RandomForest': lambda: RandomForestRegressor(random_state=42),
        'LogRandomForest': lambda: LogRandomForestRegressor(random_state=42),
        # 'LinearRegression': lambda: LinearRegression(),
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

    # Create a fresh instance of best model with train_ratio config
    best_model = MiCoProxy(best_model_factory(), preprocess=best_features_name, 
                           train_x=X, train_y=y)
    best_model.fit()
    return best_model

def get_mico_matmul_proxy(mico_type: str = 'small'):
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_matmul_zoo.csv',
        'matmul'
    )

def get_mico_conv2d_proxy(mico_type: str = 'small'):
    return get_proxy(
        f'benchmark_results/mico_{mico_type}_conv2d_zoo.csv',
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
    return get_proxy('benchmark_results/bitfusion_matmul_zoo.csv', 'matmul')

def get_bitfusion_conv2d_proxy():
    return get_proxy('benchmark_results/bitfusion_conv2d_zoo.csv', 'conv2d')

def get_host_matmul_proxy(opt="opt"):
    return get_proxy(
        f'benchmark_results/host_{opt}_matmul_zoo.csv',
        'matmul'
    )

def get_host_conv2d_proxy(opt="opt"):
    return get_proxy(
        f'benchmark_results/host_{opt}_conv2d_zoo.csv',
        'conv2d'
    )

# ============================================================================
# Transfer Learning Functions for MiCoProxy
# ============================================================================

def load_proxy_data(profile_dataset: str, kernel_type: str = 'matmul'):
    """
    Load and preprocess proxy training data from a CSV file.
    
    Args:
        profile_dataset: Path to the CSV file containing profiling data
        kernel_type: Type of kernel ('matmul' or 'conv2d')
    
    Returns:
        tuple: (X, y) where X is feature matrix and y is latency values
    """
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
            MACS = MACS / 16
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
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
    y = latency
    X = np.column_stack(RAW)
    return X, y


def get_transfer_proxy(source_type: str, target_type: str, kernel_type: str = 'matmul',
                       finetune_ratio: float = 0.1, strategy: str = 'combined',
                       preprocess: str = 'cbops+', seed: int = 42,
                       model_type: str = 'random_forest',
                       freeze_strategy: str = 'freeze_extractor',
                       verbose: bool = True):
    """
    Create a proxy model using transfer learning from source to target domain.
    
    This function enables training a proxy on one hardware target (e.g., mico_small)
    and fine-tuning it on another target (e.g., mico_high) with limited data.
    
    Args:
        source_type: Source domain identifier (e.g., 'mico_small', 'mico_high', 'bitfusion')
        target_type: Target domain identifier (e.g., 'mico_small', 'mico_high')
        kernel_type: Type of kernel ('matmul' or 'conv2d')
        finetune_ratio: Ratio of target data to use for fine-tuning (0.0 to 1.0)
        strategy: Fine-tuning strategy ('combined', 'target_only', 'weighted')
        preprocess: Feature preprocessing method
        seed: Random seed for reproducibility
        model_type: Type of model to use:
            - 'random_forest': LogRandomForestRegressor (default)
            - 'mlp': LogMLPRegressor (sklearn MLP with warm_start)
            - 'torch_mlp': TorchMLPRegressor (PyTorch MLP with layer freezing)
        freeze_strategy: For torch_mlp, controls transfer learning approach:
            - 'freeze_extractor': Freeze feature layers, train only head (default)
            - 'discriminative_lr': Use lower LR for pretrained layers
            - 'none': No freezing, fine-tune all layers
        verbose: Whether to print progress information
    
    Returns:
        tuple: (proxy, results) where proxy is the fine-tuned model and 
               results contains evaluation metrics
    
    Example:
        >>> # Transfer with PyTorch MLP and layer freezing
        >>> proxy, results = get_transfer_proxy(
        ...     source_type='mico_small',
        ...     target_type='mico_high', 
        ...     kernel_type='matmul',
        ...     finetune_ratio=0.1,
        ...     model_type='torch_mlp',
        ...     freeze_strategy='freeze_extractor'
        ... )
    """
    # Determine file paths based on source/target types
    def get_filepath(domain_type, kernel):
        if domain_type == 'bitfusion':
            return f'benchmark_results/bitfusion_{kernel}_zoo.csv'
        elif domain_type.startswith('mico_'):
            return f'benchmark_results/{domain_type}_{kernel}_zoo.csv'
        elif domain_type.startswith('host_'):
            return f'benchmark_results/{domain_type}_{kernel}_zoo.csv'
        else:
            # Assume it's a mico type (small, high, etc.)
            return f'benchmark_results/mico_{domain_type}_{kernel}_zoo.csv'
    
    source_path = get_filepath(source_type, kernel_type)
    target_path = get_filepath(target_type, kernel_type)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Transfer Learning: {source_type} -> {target_type} ({kernel_type})")
        print(f"Model type: {model_type}")
        if model_type == 'torch_mlp':
            print(f"Freeze strategy: {freeze_strategy}")
        print(f"{'='*80}")
    
    # Load source and target data
    X_source, y_source = load_proxy_data(source_path, kernel_type)
    X_target, y_target = load_proxy_data(target_path, kernel_type)
    
    if verbose:
        print(f"Source data: {len(X_source)} samples from {source_path}")
        print(f"Target data: {len(X_target)} samples from {target_path}")
    
    # Create model based on model_type
    if model_type == 'random_forest':
        model = LogRandomForestRegressor(random_state=seed)
    elif model_type == 'mlp':
        model = LogMLPRegressor(random_state=seed, hidden_layer_sizes=(64, 32))
    elif model_type == 'torch_mlp':
        model = TorchMLPRegressor(
            hidden_dims=(128, 64, 32),
            epochs=500,
            lr=0.001,
            freeze_strategy=freeze_strategy,
            random_state=seed,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest', 'mlp', or 'torch_mlp'.")
    
    proxy = MiCoProxy(model, preprocess=preprocess, seed=seed)
    proxy._source_domain = source_type
    proxy._model_type = model_type
    
    if verbose:
        print(f"\nPretraining on {source_type}...")
    proxy.pretrain(X_source, y_source)
    
    # Evaluate pretrained model on target domain (before fine-tuning)
    y_pred_before = proxy.predict(X_target)
    mape_before = mean_absolute_percentage_error(y_target, y_pred_before)
    r2_before = r2_score(y_target, y_pred_before)
    
    if verbose:
        print(f"Before fine-tuning - MAPE: {mape_before*100:.2f}%, R2: {r2_before:.4f}")
    
    # Fine-tune on target domain
    if verbose:
        print(f"\nFine-tuning on {target_type} with {finetune_ratio*100:.1f}% data...")
    finetune_results = proxy.finetune(X_target, y_target, finetune_ratio, strategy)
    
    # Evaluate after fine-tuning
    y_pred_after = proxy.predict(X_target)
    mape_after = mean_absolute_percentage_error(y_target, y_pred_after)
    r2_after = r2_score(y_target, y_pred_after)
    mae_after = mean_absolute_error(y_target, y_pred_after)
    
    if verbose:
        print(f"After fine-tuning  - MAPE: {mape_after*100:.2f}%, R2: {r2_after:.4f}")
        improvement = (mape_before - mape_after) / mape_before * 100 if mape_before > 0 else 0
        print(f"MAPE Improvement: {improvement:.1f}%")
        print(f"{'='*80}")
    
    results = {
        'source_type': source_type,
        'target_type': target_type,
        'kernel_type': kernel_type,
        'model_type': model_type,
        'finetune_ratio': finetune_ratio,
        'strategy': strategy,
        'target_samples_used': finetune_results['target_samples_used'],
        'mape_before': mape_before,
        'r2_before': r2_before,
        'mape_after': mape_after,
        'r2_after': r2_after,
        'mae_after': mae_after,
        'mape_improvement': (mape_before - mape_after) / mape_before if mape_before > 0 else 0
    }
    
    return proxy, results


def evaluate_transfer_learning(source_type: str, target_type: str, 
                               kernel_type: str = 'matmul',
                               ratios: list = None,
                               strategies: list = None,
                               n_trials: int = 5,
                               seed: int = 42,
                               verbose: bool = True):
    """
    Systematically evaluate transfer learning across different data ratios and strategies.
    
    This function runs experiments to understand:
    1. How much target data is needed for accurate prediction
    2. Which fine-tuning strategy works best
    3. The effectiveness of transfer learning vs training from scratch
    
    Args:
        source_type: Source domain identifier
        target_type: Target domain identifier
        kernel_type: Type of kernel ('matmul' or 'conv2d')
        ratios: List of fine-tuning ratios to test (default: [0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
        strategies: List of strategies to test (default: ['combined', 'target_only', 'weighted'])
        n_trials: Number of trials with different random seeds
        seed: Base random seed
        verbose: Whether to print progress
    
    Returns:
        dict: Results containing metrics for each ratio/strategy combination
    """
    if ratios is None:
        ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    if strategies is None:
        strategies = ['combined', 'target_only', 'weighted']
    
    results = {
        'source_type': source_type,
        'target_type': target_type,
        'kernel_type': kernel_type,
        'experiments': []
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Transfer Learning Evaluation: {source_type} -> {target_type} ({kernel_type})")
        print(f"Ratios: {ratios}")
        print(f"Strategies: {strategies}")
        print(f"Trials per combination: {n_trials}")
        print(f"{'='*80}")
    
    for strategy in strategies:
        for ratio in ratios:
            trial_mapes = []
            trial_r2s = []
            
            for trial in range(n_trials):
                trial_seed = seed + trial
                _, trial_results = get_transfer_proxy(
                    source_type=source_type,
                    target_type=target_type,
                    kernel_type=kernel_type,
                    finetune_ratio=ratio,
                    strategy=strategy,
                    seed=trial_seed,
                    verbose=False
                )
                trial_mapes.append(trial_results['mape_after'])
                trial_r2s.append(trial_results['r2_after'])
            
            exp_result = {
                'strategy': strategy,
                'ratio': ratio,
                'mean_mape': np.mean(trial_mapes),
                'std_mape': np.std(trial_mapes),
                'mean_r2': np.mean(trial_r2s),
                'std_r2': np.std(trial_r2s)
            }
            results['experiments'].append(exp_result)
            
            if verbose:
                print(f"  [{strategy:12s}] ratio={ratio:.2f}: "
                      f"MAPE={exp_result['mean_mape']*100:6.2f}% ± {exp_result['std_mape']*100:4.2f}%, "
                      f"R2={exp_result['mean_r2']:.4f}")
    
    return results


def compare_transfer_directions(kernel_type: str = 'matmul',
                                finetune_ratio: float = 0.1,
                                seed: int = 42,
                                verbose: bool = True):
    """
    Compare transfer learning in both directions between mico_small and mico_high.
    
    This function investigates whether transfer learning is symmetric or if
    one direction is more effective than the other.
    
    Args:
        kernel_type: Type of kernel ('matmul' or 'conv2d')
        finetune_ratio: Ratio of target data to use
        seed: Random seed
        verbose: Whether to print results
    
    Returns:
        dict: Comparison results for both directions
    """
    results = {
        'kernel_type': kernel_type,
        'finetune_ratio': finetune_ratio,
        'directions': {}
    }
    
    # small -> high
    _, small_to_high = get_transfer_proxy(
        source_type='mico_small',
        target_type='mico_high',
        kernel_type=kernel_type,
        finetune_ratio=finetune_ratio,
        seed=seed,
        verbose=verbose
    )
    results['directions']['small_to_high'] = small_to_high
    
    # high -> small
    _, high_to_small = get_transfer_proxy(
        source_type='mico_high',
        target_type='mico_small',
        kernel_type=kernel_type,
        finetune_ratio=finetune_ratio,
        seed=seed,
        verbose=verbose
    )
    results['directions']['high_to_small'] = high_to_small
    
    if verbose:
        print(f"\n{'='*80}")
        print("TRANSFER DIRECTION COMPARISON")
        print(f"{'='*80}")
        print(f"small->high: MAPE={small_to_high['mape_after']*100:.2f}%, Improvement={small_to_high['mape_improvement']*100:.1f}%")
        print(f"high->small: MAPE={high_to_small['mape_after']*100:.2f}%, Improvement={high_to_small['mape_improvement']*100:.1f}%")
    
    return results


def explore_cross_target_transfer(source_type: str = 'bitfusion',
                                  target_types: list = None,
                                  kernel_type: str = 'matmul',
                                  finetune_ratios: list = None,
                                  seed: int = 42,
                                  verbose: bool = True):
    """
    Explore transfer learning from one hardware target to another.
    
    This is useful for investigating whether knowledge from one accelerator
    (e.g., BitFusion) can be transferred to another (e.g., VexiiRiscv/MiCo).
    
    Args:
        source_type: Source hardware type
        target_types: List of target types to evaluate
        kernel_type: Kernel type
        finetune_ratios: Ratios to test
        seed: Random seed
        verbose: Whether to print results
    
    Returns:
        dict: Cross-target transfer results
    """
    if target_types is None:
        target_types = ['mico_small', 'mico_high']
    if finetune_ratios is None:
        finetune_ratios = [0.05, 0.1, 0.2, 0.5]
    
    results = {
        'source_type': source_type,
        'kernel_type': kernel_type,
        'targets': {}
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Cross-Target Transfer: {source_type} -> Multiple Targets ({kernel_type})")
        print(f"{'='*80}")
    
    for target in target_types:
        target_results = []
        
        for ratio in finetune_ratios:
            _, transfer_result = get_transfer_proxy(
                source_type=source_type,
                target_type=target,
                kernel_type=kernel_type,
                finetune_ratio=ratio,
                seed=seed,
                verbose=False
            )
            target_results.append({
                'ratio': ratio,
                'mape_before': transfer_result['mape_before'],
                'mape_after': transfer_result['mape_after'],
                'improvement': transfer_result['mape_improvement']
            })
            
            if verbose:
                print(f"  {source_type} -> {target} (ratio={ratio:.2f}): "
                      f"MAPE={transfer_result['mape_after']*100:.2f}% "
                      f"(improvement: {transfer_result['mape_improvement']*100:.1f}%)")
        
        results['targets'][target] = target_results
    
    return results


def compare_model_types_for_transfer(source_type: str = 'mico_small',
                                      target_type: str = 'mico_high',
                                      kernel_type: str = 'matmul',
                                      finetune_ratios: list = None,
                                      model_types: list = None,
                                      n_trials: int = 3,
                                      seed: int = 42,
                                      verbose: bool = True):
    """
    Compare different model types for transfer learning effectiveness.
    
    This function runs experiments comparing tree-based models (Random Forest),
    sklearn MLP, and PyTorch MLP with layer freezing for transfer learning.
    
    Args:
        source_type: Source domain identifier
        target_type: Target domain identifier
        kernel_type: Kernel type ('matmul' or 'conv2d')
        finetune_ratios: List of fine-tuning ratios to test
        model_types: List of model types to compare. Options:
            - 'random_forest': LogRandomForestRegressor
            - 'mlp': sklearn MLPRegressor with warm_start
            - 'torch_mlp': PyTorch MLP with layer freezing
        n_trials: Number of trials for averaging
        seed: Base random seed
        verbose: Whether to print results
    
    Returns:
        dict: Comparison results for all model types
    """
    if finetune_ratios is None:
        finetune_ratios = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    if model_types is None:
        model_types = ['random_forest', 'mlp', 'torch_mlp']
    
    results = {
        'source_type': source_type,
        'target_type': target_type,
        'kernel_type': kernel_type,
        'comparisons': {model_type: [] for model_type in model_types}
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON: {', '.join(model_types)} for Transfer Learning")
        print(f"Transfer: {source_type} -> {target_type} ({kernel_type})")
        print(f"{'='*80}")
    
    for ratio in finetune_ratios:
        if verbose:
            print(f"\nFine-tune ratio: {ratio*100:.0f}%")
        
        for model_type in model_types:
            trial_mapes_before = []
            trial_mapes_after = []
            trial_improvements = []
            
            for trial in range(n_trials):
                trial_seed = seed + trial
                _, result = get_transfer_proxy(
                    source_type=source_type,
                    target_type=target_type,
                    kernel_type=kernel_type,
                    finetune_ratio=ratio,
                    model_type=model_type,
                    seed=trial_seed,
                    verbose=False
                )
                trial_mapes_before.append(result['mape_before'])
                trial_mapes_after.append(result['mape_after'])
                trial_improvements.append(result['mape_improvement'])
            
            mean_mape_before = np.mean(trial_mapes_before)
            mean_mape_after = np.mean(trial_mapes_after)
            std_mape_after = np.std(trial_mapes_after)
            mean_improvement = np.mean(trial_improvements)
            
            results['comparisons'][model_type].append({
                'ratio': ratio,
                'mape_before': mean_mape_before,
                'mape_after': mean_mape_after,
                'std_mape_after': std_mape_after,
                'improvement': mean_improvement
            })
            
            if verbose:
                print(f"  {model_type:15s}: MAPE={mean_mape_after*100:6.2f}% ± {std_mape_after*100:4.2f}% "
                      f"(improvement: {mean_improvement*100:5.1f}%)")
    
    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY: Which model is better for transfer learning?")
        print(f"{'='*80}")
        
        win_counts = {m: 0 for m in model_types}
        
        for i, ratio in enumerate(finetune_ratios):
            mapes = {m: results['comparisons'][m][i]['mape_after'] for m in model_types}
            best_model = min(mapes, key=mapes.get)
            win_counts[best_model] += 1
            
            best_mape = mapes[best_model]
            print(f"  Ratio {ratio*100:3.0f}%: {best_model} wins (MAPE={best_mape*100:.2f}%)")
        
        print(f"\nWin counts: {win_counts}")
        best_overall = max(win_counts, key=win_counts.get)
        print(f"Recommendation: Use {best_overall} (model_type='{best_overall}') for best overall performance.")
    
    return results



if __name__ == "__main__":
    # Test Bitfusion proxies with cross-validation
    # print("\n" + "="*80)
    # print("BITFUSION PROXY TUNING")
    # print("="*80)
    # matmul_proxy = get_bitfusion_matmul_proxy()
    # conv2d_proxy = get_bitfusion_conv2d_proxy()
    
    # Test MiCo proxies with cross-validation
    print("\n" + "="*80)
    print("MICO PROXY TUNING")
    print("="*80)
    
    # # Test for 'small' mico type
    print("\n### Testing MiCo 'small' type ###")
    mico_small_matmul_proxy = get_mico_matmul_proxy(mico_type='small')
    mico_small_conv2d_proxy = get_mico_conv2d_proxy(mico_type='small')
    
    # # # Test for 'high' mico type
    # print("\n### Testing MiCo 'high' type ###")
    # mico_high_matmul_proxy = get_mico_matmul_proxy(mico_type='high')
    # mico_high_conv2d_proxy = get_mico_conv2d_proxy(mico_type='high')
    
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