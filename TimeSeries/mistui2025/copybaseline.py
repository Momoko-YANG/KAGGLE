"""
MITSUI COMMODITY PREDICTION - ADVANCED MODEL V3 WITH LSTM
==========================================================
Enhances V2 model with LSTM feature extraction for capturing deep temporal patterns

KEY ENHANCEMENTS OVER V2:
1. LSTM feature extractor for sequential pattern learning
2. Hybrid approach: LSTM features + traditional features
3. Multi-scale sequence processing (different lookback windows)
4. Attention mechanism for important time steps
5. Ensemble of LSTM architectures for robustness
"""

import pandas as pd
import polars as pl
import numpy as np
import warnings
import gc
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
warnings.simplefilter('ignore')

# Core ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Optional: LightGBM for advanced modeling
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available, using RandomForest fallback")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    """Enhanced configuration with LSTM parameters"""
    # Paths
    path: str = "/kaggle/input/mitsui-commodity-prediction-challenge/"
    seed: int = 42
    
    # Data splits
    train_end: int = 1826
    test_start: int = 1827
    test_end: int = 1956
    
    # Feature engineering
    windows: Tuple[int,...] = (5, 10, 20, 60)
    ewma_span: int = 10
    corr_window: int = 20
    
    # Advanced features
    use_pca: bool = True
    pca_components: int = 5
    use_clustering: bool = True
    n_clusters: int = 8
    
    # LSTM parameters
    use_lstm_features: bool = True
    lstm_sequence_lengths: Tuple[int,...] = (15, 30, 60)  # Multiple lookback windows
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_feature_dim: int = 32  # Output dimension of LSTM features
    lstm_batch_size: int = 128
    lstm_epochs: int = 20
    lstm_learning_rate: float = 0.001
    use_attention: bool = True
    
    # Model training
    n_folds: int = 3
    n_features_per_lag: int = 80  # Increased to accommodate LSTM features
    use_two_stage: bool = True
    
    # Other
    solution_null_filler: float = 0.0
    targets: List[str] = field(default_factory=lambda: [f"target_{i}" for i in range(424)])
    
    # Device for PyTorch
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ==============================================================================
# LSTM COMPONENTS
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for LSTM training"""
    def __init__(self, sequences, targets=None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]

class AttentionLayer(nn.Module):
    """Simple attention mechanism for LSTM"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output, attention_weights

class LSTMFeatureExtractor(nn.Module):
    """LSTM model for extracting features from time series"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 output_size=32, dropout=0.2, use_attention=True):
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        if self.use_attention:
            # Apply attention
            attended_out, _ = self.attention(lstm_out)
        else:
            # Use last timestep
            attended_out = lstm_out[:, -1, :]
        
        # Pass through FC layers
        x = F.relu(self.fc1(attended_out))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        
        return x

class LSTMFeatureEngineering:
    """LSTM-based feature engineering"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}  # Store models for different sequence lengths
        self.scalers = {}  # Store scalers for normalization
        self.fitted = False
        
    def prepare_sequences(self, df, sequence_length, feature_cols):
        """Prepare sequences for LSTM input"""
        sequences = []
        indices = []
        
        # Sort by date to ensure temporal order
        df = df.sort_values('date_id')
        
        for i in range(sequence_length, len(df)):
            # Extract sequence of features
            seq = df[feature_cols].iloc[i-sequence_length:i].values
            sequences.append(seq)
            indices.append(i)
            
        return np.array(sequences), indices
    
    def get_base_features(self, df):
        """Get base features for LSTM input (prices, volumes, returns)"""
        base_features = []
        
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'open', 'high', 'low'])]
        vol_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        # Identify all columns needed for feature generation
        cols_to_use = price_cols[:20] + vol_cols[:10]
        
        # Create a safe, numeric-only copy of the required data
        local_df = df[['date_id'] + cols_to_use].copy()
        for col in cols_to_use:
            if local_df[col].dtype == 'object':
                local_df[col] = pd.to_numeric(local_df[col], errors='coerce')

        # Add the original base feature names
        base_features.extend(price_cols[:20])
        base_features.extend(vol_cols[:10])
        
        # Calculate return features on the clean, local DataFrame
        for col in price_cols[:20]:
            ret_col = f'{col}_return'
            if ret_col not in df.columns: # Check original df to prevent re-adding
                # 推荐写法（禁用填充 + 提升精度再降回）
                s = local_df[col]
                # 强制用 float64 计算，避免底层 pad/backfill 对 float16 的坑
                ret = s.astype('float64').pct_change(fill_method=None)
                # 可选：把极值/非有限数清掉
                ret = ret.replace([np.inf, -np.inf], np.nan)
                
                # 写回到两个 DataFrame
                local_df[ret_col] = ret.astype('float32')   # 节省内存
                df[ret_col]       = local_df[ret_col]

            base_features.append(ret_col)
        
        # Final check and limit
        base_features = [col for col in base_features if col in df.columns]
        return base_features[:50]
    
    def fit(self, train_df, train_labels=None):
        """Fit LSTM models on training data"""
        print("Training LSTM feature extractors...")
        
        # Get base features
        base_features = self.get_base_features(train_df)
        input_size = len(base_features)
        
        # Train a model for each sequence length
        for seq_len in self.config.lstm_sequence_lengths:
            print(f"  Training LSTM for sequence length {seq_len}...")
            
            # Prepare sequences
            sequences, indices = self.prepare_sequences(train_df, seq_len, base_features)
            
            if len(sequences) < 100:
                print(f"    Skipping seq_len {seq_len} - insufficient data")
                continue
            
            # Normalize sequences
            scaler = StandardScaler()
            sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
            sequences_normalized = scaler.fit_transform(sequences_reshaped)
            sequences_normalized = sequences_normalized.reshape(sequences.shape)
            self.scalers[seq_len] = scaler
            
            # Prepare targets (if available, use for supervised pre-training)
            targets = None
            if train_labels is not None:
                # Use mean of next day's targets as supervision signal
                valid_indices = [idx for idx in indices if idx < len(train_labels)]
                if valid_indices:
                    target_cols = [col for col in train_labels.columns if 'target' in col][:10]
                    targets = train_labels[target_cols].iloc[valid_indices].mean(axis=1).values
            
            # Create model
            model = LSTMFeatureExtractor(
                input_size=input_size,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                output_size=self.config.lstm_feature_dim,
                dropout=self.config.lstm_dropout,
                use_attention=self.config.use_attention
            ).to(self.config.device)
            
            # Train model
            self._train_lstm(model, sequences_normalized, targets)
            
            # Store model
            self.models[seq_len] = model
        
        self.fitted = True
        self.base_features = base_features
        return self
    
    def _train_lstm(self, model, sequences, targets=None):
        """Train a single LSTM model"""
        # Create dataset
        if targets is not None:
            # Supervised training
            dataset = TimeSeriesDataset(sequences, targets)
            criterion = nn.MSELoss()
        else:
            # Unsupervised training (autoencoder-style)
            # Use last timestep prediction as target
            targets = sequences[:, -1, :].mean(axis=1)  # Simple target
            dataset = TimeSeriesDataset(sequences, targets)
            criterion = nn.MSELoss()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.lstm_batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.lstm_learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        # Training loop
        model.train()
        for epoch in range(self.config.lstm_epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                # Forward pass
                features = model(batch_x)
                
                # For unsupervised, predict some property of the sequence
                # Here we use a simple projection to match target dimension
                if features.shape[1] != batch_y.shape[-1]:
                    projection = nn.Linear(features.shape[1], batch_y.shape[-1]).to(self.config.device)
                    features = projection(features)
                
                loss = criterion(features.squeeze(), batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}/{self.config.lstm_epochs}, Loss: {avg_loss:.4f}")
    
    def transform(self, df):
        """Extract LSTM features from data"""
        if not self.fitted:
            raise ValueError("LSTM models not fitted yet")
        
        lstm_features = pd.DataFrame(index=df.index)
        
        # Extract features for each sequence length
        for seq_len, model in self.models.items():
            # Prepare sequences
            sequences, indices = self.prepare_sequences(df, seq_len, self.base_features)
            
            if len(sequences) == 0:
                continue
            
            # Normalize
            scaler = self.scalers[seq_len]
            sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
            sequences_normalized = scaler.transform(sequences_reshaped)
            sequences_normalized = sequences_normalized.reshape(sequences.shape)
            
            # Extract features
            model.eval()
            with torch.no_grad():
                sequences_tensor = torch.FloatTensor(sequences_normalized).to(self.config.device)
                features = model(sequences_tensor).cpu().numpy()
            
            # Create feature dataframe
            for i in range(features.shape[1]):
                feature_name = f'lstm_seq{seq_len}_feat{i}'
                # Initialize with NaN
                lstm_features[feature_name] = np.nan
                # Fill valid indices
                lstm_features.loc[df.index[indices], feature_name] = features[:, i]
        
        # Forward fill NaN values for early time steps
        lstm_features = lstm_features.fillna(method='bfill').fillna(0)
        
        return lstm_features

# ==============================================================================
# EXISTING FUNCTIONS (from V2, keeping as is)
# ==============================================================================

def reduce_mem_usage(df):
    """Reduce memory usage of dataframe"""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def load_data():
    """Load and preprocess all data files"""
    print("📂 Loading data...")
    
    train = pd.read_csv(CFG.path + "train.csv").sort_values("date_id")
    train = reduce_mem_usage(train)
    
    train_labels = pd.read_csv(CFG.path + "train_labels.csv")
    
    # Fix column names
    label_cols = [col for col in train_labels.columns if col != 'date_id']
    rename_map = {col: f"target_{col}" for col in label_cols}
    train_labels = train_labels.rename(columns=rename_map)
    print("Renamed train_labels columns to target_X format.")
    
    train_labels = reduce_mem_usage(train_labels)
    
    target_pairs = pd.read_csv(CFG.path + "target_pairs.csv")
    
    print(f"Loaded: train {train.shape}, labels {train_labels.shape}, pairs {len(target_pairs)}")
    return train, train_labels, target_pairs

def preprocess_columns(df):
    """
    通用版本：检查所有列，如果某列是'object'类型，
    则尝试将其转换为数字，无法转换的值会变成NaN。
    """
    df = df.copy()
    # 遍历除 date_id 之外的所有列
    for col in df.columns:
        if col == 'date_id':
            continue
        
        # 如果列的数据类型是 'object'
        if df[col].dtype == 'object':
            print(f"  Fixing object-type column: {col}")
            # 强制转换为数字类型，任何无法转换的都会变成 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def parse_pairs_advanced(train_df, target_pairs):
    """Parse pairs and create spread definitions"""
    target_to_lag = {}
    spread_definitions = []
    
    for _, row in target_pairs.iterrows():
        target_id = row['target']
        target_name = f"target_{target_id}"
        lag = row['lag']
        target_to_lag[target_name] = lag
        
        # Parse spread definition
        if pd.notna(row['pair']):
            pair_str = row['pair'].strip()
            if ' - ' in pair_str:
                # It's a spread
                parts = pair_str.split(' - ')
                col_a = parts[0].strip()
                col_b = parts[1].strip()
                if col_a in train_df.columns and col_b in train_df.columns:
                    spread_definitions.append({
                        'target': target_name,
                        'lag': lag,
                        'col_a': col_a,
                        'col_b': col_b,
                        'is_spread': True
                    })
            else:
                # Single asset
                if pair_str in train_df.columns:
                    spread_definitions.append({
                        'target': target_name,
                        'lag': lag,
                        'col_a': pair_str,
                        'col_b': None,
                        'is_spread': False
                    })
    
    spreads_df = pd.DataFrame(spread_definitions)
    return target_to_lag, spreads_df

# [Keep all existing classes: AdvancedFeatureEngineer, TwoStageModel, etc.]
# [For brevity, I'm not repeating them here, but they remain unchanged from V2]

class AdvancedFeatureEngineer:
    """Advanced feature engineering with spreads and cross-sectional features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pca_model = None
        self.kmeans_model = None
        self.fitted = False
        
    def fit(self, train_df, spreads_df):
        """Fit PCA and clustering models"""
        spreads = self._create_spreads(train_df, spreads_df)
        
        if self.config.use_pca and len(spreads.columns) > 10:
            self.pca_model = PCA(n_components=min(self.config.pca_components, len(spreads.columns)-1))
            self.pca_model.fit(spreads.fillna(0))
        
        if self.config.use_clustering and len(spreads.columns) > 10:
            self.kmeans_model = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.seed)
            if self.pca_model is not None:
                pca_features = self.pca_model.transform(spreads.fillna(0))
                self.kmeans_model.fit(pca_features)
            else:
                self.kmeans_model.fit(spreads.fillna(0))
        
        self.fitted = True
        return self
    
    def transform(self, df, spreads_df):
        """Create all features"""
        df = df.copy()
        
        spreads = self._create_spreads(df, spreads_df)
        spread_features = self._create_spread_features(spreads)
        cross_features = self._create_cross_sectional_features(spreads)
        
        if self.fitted and self.pca_model is not None:
            pca_features = pd.DataFrame(
                self.pca_model.transform(spreads.fillna(0)),
                columns=[f'pca_{i}' for i in range(self.pca_model.n_components_)],
                index=spreads.index
            )
            df = pd.concat([df, pca_features], axis=1)
        
        if self.fitted and self.kmeans_model is not None:
            if self.pca_model is not None:
                pca_features = self.pca_model.transform(spreads.fillna(0))
                clusters = self.kmeans_model.predict(pca_features)
            else:
                clusters = self.kmeans_model.predict(spreads.fillna(0))
            
            df['cluster_id'] = clusters
            for i in range(self.config.n_clusters):
                df[f'cluster_{i}'] = (clusters == i).astype(int)
        
        df = pd.concat([df, spread_features, cross_features], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df
    
    def _create_spreads(self, df, spreads_df):
        """Create spread series from definitions"""
        spreads = pd.DataFrame(index=df.index)
        
        for _, row in spreads_df.iterrows():
            target_name = row['target']
            if row['is_spread']:
                spreads[f"spread_{target_name}"] = (
                    np.log(df[row['col_a']] + 1e-10) - 
                    np.log(df[row['col_b']] + 1e-10)
                )
            else:
                spreads[f"spread_{target_name}"] = np.log(df[row['col_a']] + 1e-10)
        
        return spreads
    
    def _create_spread_features(self, spreads):
        """Create time-series features on spreads"""
        features = pd.DataFrame(index=spreads.index)
        
        for col in spreads.columns:
            features[f'{col}_ret1'] = spreads[col].diff()
            
            for window in self.config.windows:
                features[f'{col}_ma{window}'] = spreads[col].rolling(window, min_periods=1).mean()
                features[f'{col}_vol{window}'] = features[f'{col}_ret1'].rolling(window, min_periods=1).std()
                
                ma = features[f'{col}_ma{window}']
                vol = features[f'{col}_vol{window}']
                features[f'{col}_zscore{window}'] = (spreads[col] - ma) / (vol + 1e-10)
            
            features[f'{col}_ewma'] = spreads[col].ewm(span=self.config.ewma_span, adjust=False).mean()
            features[f'{col}_skew20'] = features[f'{col}_ret1'].rolling(20, min_periods=1).skew()
            features[f'{col}_kurt20'] = features[f'{col}_ret1'].rolling(20, min_periods=1).kurt()
        
        return features
    
    def _create_cross_sectional_features(self, spreads):
        """Create cross-sectional features"""
        features = pd.DataFrame(index=spreads.index)
        
        for col in spreads.columns[:20]:
            features[f'{col}_xrank'] = spreads.groupby(level=0)[col].rank(pct=True, method='average')
            features[f'{col}_xz'] = spreads.groupby(level=0)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
        
        features['market_mean'] = spreads.mean(axis=1)
        features['market_std'] = spreads.std(axis=1)
        features['market_skew'] = spreads.skew(axis=1)
        
        return features

def make_time_folds(dates, n_folds=3):
    """Create time-series cross-validation folds"""
    unique_dates = sorted(dates.unique())
    n = len(unique_dates)
    fold_size = max(1, n // n_folds)
    
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds - 1 else n
        val_dates = set(unique_dates[val_start:val_end])
        train_dates = set(unique_dates[:val_start])
        
        if len(train_dates) > 0 and len(val_dates) > 0:
            folds.append((train_dates, val_dates))
    
    return folds

def select_features_by_cv(X, y, dates, feature_names, n_keep=64, n_folds=3, seed=42):
    """Feature selection using time-series CV"""
    print(f"Selecting top {n_keep} features from {len(feature_names)} using {n_folds}-fold CV...")
    
    l1_importance = np.zeros(len(feature_names))
    tree_importance = np.zeros(len(feature_names))
    
    folds = make_time_folds(dates, n_folds)
    
    for fold_idx, (train_dates, val_dates) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{len(folds)}")
        
        train_mask = dates.isin(train_dates)
        val_mask = dates.isin(val_dates)
        
        if train_mask.sum() < 100 or val_mask.sum() < 50:
            continue
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train_mean = y_train.mean(axis=1)
        else:
            y_train_mean = y_train.flatten()
        
        try:
            lr = make_pipeline(
                SimpleImputer(strategy='constant', fill_value=0),
                StandardScaler(),
                LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=500, random_state=seed)
            )
            lr.fit(X_train, (y_train_mean > 0).astype(int))
            coef = np.abs(lr.named_steps['logisticregression'].coef_.flatten())
            l1_importance += coef / len(folds)
        except:
            pass
        
        if HAS_LIGHTGBM:
            try:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=31, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbosity=-1
                )
                lgb_model.fit(X_train, y_train_mean)
                tree_importance += lgb_model.feature_importances_ / len(folds)
            except:
                rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=seed, n_jobs=-1)
                rf.fit(X_train, y_train_mean)
                tree_importance += rf.feature_importances_ / len(folds)
        else:
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_train_mean)
            tree_importance += rf.feature_importances_ / len(folds)
    
    combined_importance = 0.4 * (l1_importance / (l1_importance.max() + 1e-10)) + \
                         0.6 * (tree_importance / (tree_importance.max() + 1e-10))
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': combined_importance,
        'l1_imp': l1_importance,
        'tree_imp': tree_importance
    }).sort_values('importance', ascending=False)
    
    selected = importance_df.head(n_keep)['feature'].tolist()
    print(f"  Top features: {selected[:5]}")
    
    return selected, importance_df

class TwoStageModel:
    """Two-stage model: LogisticRegression for direction + LightGBM for magnitude"""
    
    def __init__(self, seed=42):
        self.seed = seed
        self.direction_models = []
        self.magnitude_models = []
        self.calibrator = None
        self.fitted = False
    
    def fit(self, X, y, dates, n_folds=3):
        """Train using time-series cross-validation"""
        folds = make_time_folds(dates, n_folds)
        
        oof_direction = np.full(len(y), np.nan)
        oof_magnitude = np.full(len(y), np.nan)
        
        for fold_idx, (train_dates, val_dates) in enumerate(folds):
            print(f"    Training fold {fold_idx + 1}/{len(folds)}")
            
            train_mask = dates.isin(train_dates)
            val_mask = dates.isin(val_dates)
            
            if train_mask.sum() < 100:
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val = X[val_mask]
            
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train = y_train.mean(axis=1)
            
            # Train direction model
            dir_model = make_pipeline(
                SimpleImputer(strategy='constant', fill_value=0),
                StandardScaler(),
                LogisticRegression(penalty='l2', C=1.0, max_iter=500, random_state=self.seed)
            )
            dir_model.fit(X_train, (y_train > 0).astype(int))
            self.direction_models.append(dir_model)
            
            # Train magnitude model
            if HAS_LIGHTGBM:
                mag_model = lgb.LGBMRegressor(
                    n_estimators=200, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=self.seed, verbosity=-1
                )
            else:
                mag_model = RandomForestRegressor(
                    n_estimators=100, max_depth=15,
                    random_state=self.seed, n_jobs=-1
                )
            
            mag_model.fit(X_train, np.abs(y_train))
            self.magnitude_models.append(mag_model)
            
            # Out-of-fold predictions
            if val_mask.sum() > 0:
                oof_direction[val_mask] = dir_model.predict_proba(X_val)[:, 1]
                oof_magnitude[val_mask] = mag_model.predict(X_val)
        
        # Calibrate predictions using isotonic regression
        valid_mask = ~(np.isnan(oof_direction) | np.isnan(oof_magnitude))
        if valid_mask.sum() > 100:
            oof_combined = (2 * oof_direction[valid_mask] - 1) * oof_magnitude[valid_mask]
            y_valid = y[valid_mask]
            if len(y_valid.shape) > 1:
                y_valid = y_valid.mean(axis=1)
            
            self.calibrator = IsotonicRegression(y_min=-5, y_max=5, out_of_bounds='clip')
            self.calibrator.fit(oof_combined, y_valid)
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using ensemble of models"""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        direction_preds = []
        magnitude_preds = []
        
        for dir_model, mag_model in zip(self.direction_models, self.magnitude_models):
            direction_preds.append(dir_model.predict_proba(X)[:, 1])
            magnitude_preds.append(mag_model.predict(X))
        
        direction = np.mean(direction_preds, axis=0)
        magnitude = np.mean(magnitude_preds, axis=0)
        combined = (2 * direction - 1) * magnitude
        
        if self.calibrator is not None:
            combined = self.calibrator.predict(combined)
        
        return combined

# ==============================================================================
# ENHANCED TRAINING PIPELINE WITH LSTM
# ==============================================================================
def train_advanced_models_with_lstm():
    """Main training pipeline with LSTM and traditional features"""
    print("="*70)
    print("MITSUI ADVANCED MODEL WITH LSTM TRAINING")
    print("="*70)
    
    # Load data
    train, train_labels, target_pairs = load_data()
    train = preprocess_columns(train)
    
    # Parse pairs with spread definitions
    target_to_lag, spreads_df = parse_pairs_advanced(train, target_pairs)
    print(f"Parsed {len(spreads_df)} spread definitions")
    
    # === NEW: LSTM Feature Engineering ===
    lstm_features_df = None
    if CFG.use_lstm_features:
        print("\n🧠 Training LSTM feature extractors...")
        lstm_fe = LSTMFeatureEngineering(CFG)
        lstm_fe.fit(train, train_labels)
        lstm_features_df = lstm_fe.transform(train)
        print(f"Generated {len(lstm_features_df.columns)} LSTM features")
        
        # Combine with original dataframe
        train = pd.concat([train, lstm_features_df], axis=1)
        train = reduce_mem_usage(train)
    
    # Traditional feature engineering
    print("\n🔧 Creating traditional features...")
    fe = AdvancedFeatureEngineer(CFG)
    fe.fit(train, spreads_df)
    train_enhanced = fe.transform(train, spreads_df)
    train_enhanced = reduce_mem_usage(train_enhanced)
    
    # Get feature columns (now includes LSTM features)
    feature_cols = [c for c in train_enhanced.columns 
                   if c not in ["date_id"] + CFG.targets]
    print(f"Total features (including LSTM): {len(feature_cols)}")
    
    # Split data
    X_train_all = train_enhanced.iloc[:CFG.train_end+1][feature_cols].fillna(0).values
    X_test_all = train_enhanced.iloc[CFG.test_start:CFG.test_end+1][feature_cols].fillna(0).values
    dates_train = train_enhanced.iloc[:CFG.train_end+1]['date_id']
    dates_test = train_enhanced.iloc[CFG.test_start:CFG.test_end+1]['date_id']
    
    # Store models for each lag
    models_by_lag = {}
    features_by_lag = {}
    
    # Train separate model for each lag
    for lag in [1, 2, 3, 4]:
        print(f"\n{'='*50}")
        print(f"TRAINING LAG {lag} MODEL")
        print('='*50)
        
        # Get targets for this lag
        targets_for_lag = [t for t, l in target_to_lag.items() if l == lag]
        if not targets_for_lag:
            print(f"No targets for lag {lag}, skipping...")
            continue
        
        print(f"Found {len(targets_for_lag)} targets for lag {lag}")
        
        # Get labels
        y_train = train_labels.iloc[:CFG.train_end+1][targets_for_lag].fillna(CFG.solution_null_filler).values
        y_test = train_labels.iloc[CFG.test_start:CFG.test_end+1][targets_for_lag].fillna(CFG.solution_null_filler).values
        
        # Feature selection using CV (now includes LSTM features)
        selected_features, importance_df = select_features_by_cv(
            X_train_all, y_train, dates_train,
            feature_cols, n_keep=CFG.n_features_per_lag,
            n_folds=CFG.n_folds, seed=CFG.seed
        )
        
        # Check how many LSTM features were selected
        lstm_selected = [f for f in selected_features if 'lstm_' in f]
        print(f"  LSTM features selected: {len(lstm_selected)}/{len([f for f in feature_cols if 'lstm_' in f])}")
        
        # Get indices of selected features
        feature_indices = [feature_cols.index(f) for f in selected_features]
        features_by_lag[lag] = selected_features
        
        # Prepare data with selected features
        X_train = X_train_all[:, feature_indices]
        X_test = X_test_all[:, feature_indices]
        
        # Train model
        if CFG.use_two_stage:
            print(f"Training two-stage model for lag {lag}...")
            model = TwoStageModel(seed=CFG.seed)
            model.fit(X_train, y_train, dates_train, n_folds=CFG.n_folds)
        else:
            print(f"Training RandomForest for lag {lag}...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestRegressor(
                n_estimators=100, max_depth=20,
                min_samples_split=5, min_samples_leaf=2,
                random_state=CFG.seed, n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
        
        models_by_lag[lag] = model
        
        # Validate
        if CFG.use_two_stage:
            y_pred = model.predict(X_test)
        else:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled).mean(axis=1) if len(y_test.shape) > 1 else model.predict(X_test_scaled)
        
        # Simple correlation metric for validation
        if len(y_test.shape) > 1:
            y_test_flat = y_test.mean(axis=1)
        else:
            y_test_flat = y_test
        
        corr = np.corrcoef(y_test_flat, y_pred)[0, 1]
        print(f"Validation correlation for lag {lag}: {corr:.4f}")
    
    print("\n✅ All models trained successfully!")
    
    # Return enhanced model info including LSTM components
    return {
        'models_by_lag': models_by_lag,
        'features_by_lag': features_by_lag,
        'target_to_lag': target_to_lag,
        'feature_engineer': fe,
        'lstm_feature_engineer': lstm_fe if CFG.use_lstm_features else None,
        'spreads_df': spreads_df,
        'feature_cols': feature_cols
    }

# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================
TRAINED_MODELS = None

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pd.DataFrame:
    """Prediction function for Kaggle submission"""
    global TRAINED_MODELS
    
    # Train models if not already done
    if TRAINED_MODELS is None:
        TRAINED_MODELS = train_advanced_models_with_lstm()
    
    # Convert test data to pandas
    test_pd = test.to_pandas()
    test_pd = preprocess_columns(test_pd)
    
    # Apply LSTM feature engineering if available
    if TRAINED_MODELS['lstm_feature_engineer'] is not None:
        lstm_features = TRAINED_MODELS['lstm_feature_engineer'].transform(test_pd)
        test_pd = pd.concat([test_pd, lstm_features], axis=1)
    
    # Apply traditional feature engineering
    test_enhanced = TRAINED_MODELS['feature_engineer'].transform(
        test_pd, TRAINED_MODELS['spreads_df']
    )
    
    # Get all features
    X_all = test_enhanced[TRAINED_MODELS['feature_cols']].fillna(0).values
    
    # Initialize predictions
    predictions = pd.DataFrame()
    
    # Make predictions for each lag
    for lag in [1, 2, 3, 4]:
        if lag not in TRAINED_MODELS['models_by_lag']:
            continue
        
        # Get model and features for this lag
        model = TRAINED_MODELS['models_by_lag'][lag]
        selected_features = TRAINED_MODELS['features_by_lag'][lag]
        feature_indices = [TRAINED_MODELS['feature_cols'].index(f) for f in selected_features]
        
        # Get targets for this lag
        targets_for_lag = [t for t, l in TRAINED_MODELS['target_to_lag'].items() if l == lag]
        
        # Prepare features
        X = X_all[:, feature_indices]
        
        # Predict
        if CFG.use_two_stage and isinstance(model, TwoStageModel):
            y_pred = model.predict(X)
            # Expand to match number of targets if needed
            if len(targets_for_lag) > 1:
                y_pred = np.tile(y_pred.reshape(-1, 1), (1, len(targets_for_lag)))
            else:
                y_pred = y_pred.reshape(-1, 1)
        else:
            # RandomForest case
            y_pred = model.predict(X)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
        
        # Store predictions
        lag_predictions = pd.DataFrame(y_pred, columns=targets_for_lag)
        predictions = pd.concat([predictions, lag_predictions], axis=1)
    
    # Ensure all targets have predictions
    for target in CFG.targets:
        if target not in predictions.columns:
            predictions[target] = CFG.solution_null_filler
    
    # --- 新增的修复代码 ---
    # 在返回最终结果前，将所有 NaN 值填充为默认值 (e.g., 0.0)
    # 这样可以保证提交文件中没有任何空值
    final_predictions = predictions[CFG.targets].fillna(CFG.solution_null_filler)
    
    return final_predictions

# ==============================================================================
# KAGGLE SUBMISSION SERVER
# ==============================================================================
if __name__ == "__main__":
    import kaggle_evaluation.mitsui_inference_server
    
    # Create inference server
    inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)
    
    # Check if running on Kaggle or locally
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("🚀 Starting Kaggle inference server...")
        inference_server.serve()
    else:
        print("🧪 Running local test...")
        # Train models first
        TRAINED_MODELS = train_advanced_models_with_lstm()
        
        # Show summary
        print("\n" + "="*70)
        print("MODEL SUMMARY")
        print("="*70)
        for lag in [1, 2, 3, 4]:
            if lag in TRAINED_MODELS['models_by_lag']:
                model = TRAINED_MODELS['models_by_lag'][lag]
                n_features = len(TRAINED_MODELS['features_by_lag'][lag])
                model_type = "Two-Stage" if isinstance(model, TwoStageModel) else "RandomForest"
                targets = [t for t, l in TRAINED_MODELS['target_to_lag'].items() if l == lag]
                
                # Count LSTM features
                lstm_features = [f for f in TRAINED_MODELS['features_by_lag'][lag] if 'lstm_' in f]
                print(f"Lag {lag}: {model_type} model, {n_features} features ({len(lstm_features)} LSTM), {len(targets)} targets")
        
        print("\n✅ Model with LSTM features ready for predictions!")