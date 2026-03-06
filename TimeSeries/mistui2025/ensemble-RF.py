# Mitsui Clean Baseline – 修正版本
# 主要修正：
# 1. 移除数据泄漏（no lookahead bias）
# 2. 增强鲁棒性（对数处理、错误处理）
# 3. 代码清理（移除冗余代码）
# 4. 改进特征管理
# 5. 修复关键bug：汇总打印键、Stacker训练、空折处理、融合策略
# ============================================================

from __future__ import annotations
import os, sys, re
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Tuple, List, Set

# 过滤警告
warnings.filterwarnings("ignore")                 # 全局屏蔽
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# 细粒度屏蔽常见噪音
for cat in [
    UserWarning, FutureWarning, RuntimeWarning,
]:
    warnings.filterwarnings("ignore", category=cat)

# LightGBM 静默
os.environ["LIGHTGBM_VERBOSE"] = "0"

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier

# 已移除顶层重包导入；在训练/调参/特征增强函数内部按需导入并容错
# 已移除 EMD 与 SSA 依赖


# ----------------------------
# 全局可调配置
# ----------------------------
CONFIG = {
    "INPUT_DIR": "/kaggle/input/mitsui-commodity-prediction-challenge",
    "LABELS_FILLER": -999,
    "SEED": 42,

    # 缺失值策略
    "MISSING": {
        "discontinued_threshold": 0.80,     # 列缺失率≥80%且尾端全缺视作停牌
        "discontinued_strategy": "fill_last", # 'fill_last' | 'indicator' | 'fill_zero' | 'drop'
        "regular_strategy": "forward_only",   # 修改：只用前向填充，避免数据泄漏
    },

    # 特征工程
    "WINDOWS": (5, 10, 20, 60, 120, 252),
    "EWMA_SPAN": 10,
    "CORR_W": 20,                 
    "PCA_K": 4, "KMEANS_K": 12,   

    # 特征筛选 & 训练
    "N_FOLDS": 5,
    "FS_KEEP": 48,  # 减少特征数量：从64降到48，提高泛化能力                
    
    # 价格处理
    "LOG_EPSILON": 1e-8,  # 新增：对数处理时的最小值
    "LEAKY_FEATURE_LAG": 21,  # 新增：为有泄漏风险的特征设置的滞后期
    
    # 已移除混合损失模型开关
}

# ============================================================
# ★★★ 新增：将找到的最优参数固化在这里 ★★★
# ============================================================
BEST_PARAMS_FOUND = {
    1: {'logreg_C': 0.0967183550584758, 'lgbm_lr': 0.07255091881185716, 'lgbm_num_leaves': 54, 'lgbm_max_depth': 6, 'lgbm_min_data_in_leaf': 28, 'lgbm_feature_fraction': 0.874908716857276, 'lgbm_bagging_fraction': 0.6048239211524721, 'lgbm_reg_alpha': 0.12693360252308228, 'lgbm_reg_lambda': 1.8079273833769658, 'fs_keep': 48},
    2: {'logreg_C': 0.8105016126411579, 'lgbm_lr': 0.11851859527055603, 'lgbm_num_leaves': 122, 'lgbm_max_depth': 10, 'lgbm_min_data_in_leaf': 34, 'lgbm_feature_fraction': 0.8765622705069351, 'lgbm_bagging_fraction': 0.6265477506155759, 'lgbm_reg_alpha': 0.1959828624191452, 'lgbm_reg_lambda': 0.09045457782107613, 'fs_keep': 48},
    3: {'logreg_C': 0.06391687041037726, 'lgbm_lr': 0.0897932033570188, 'lgbm_num_leaves': 91, 'lgbm_max_depth': 10, 'lgbm_min_data_in_leaf': 18, 'lgbm_feature_fraction': 0.8734061811841441, 'lgbm_bagging_fraction': 0.812965266984314, 'lgbm_reg_alpha': 0.10943180441794581, 'lgbm_reg_lambda': 1.0886737454969304, 'fs_keep': 72},
    4: {'logreg_C': 0.01671971403387934, 'lgbm_lr': 0.03532026717953224, 'lgbm_num_leaves': 88, 'lgbm_max_depth': 6, 'lgbm_min_data_in_leaf': 32, 'lgbm_feature_fraction': 0.82269844090237, 'lgbm_bagging_fraction': 0.8654062784961335, 'lgbm_reg_alpha': 0.368654792956084, 'lgbm_reg_lambda': 1.1018915653243986, 'fs_keep': 32}
}
# ============================================================


# 仅在需要时导入，失败就降级/跳过对应分支
def _maybe_import_training_libs():
    try:
        import lightgbm as lgb
    except Exception:
        lgb = None
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    try:
        from catboost import CatBoostRegressor
    except Exception:
        CatBoostRegressor = None
    try:
        import optuna
    except Exception:
        optuna = None
    try:
        import pywt
    except Exception:
        pywt = None
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.seasonal import STL
    except Exception:
        sm = None
        STL = None
    return lgb, xgb, CatBoostRegressor, optuna, pywt, sm, STL

# ============================================================
# A) DataIO
# ============================================================
def read_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    assert 'date_id' in df.columns, "train.csv 缺少 date_id"
    df['date_id'] = df['date_id'].astype('int32')
    df = df.sort_values('date_id').reset_index(drop=True)
    for c in df.columns:
        if c != 'date_id':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df

def read_labels(path: str, filler: float|int) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    assert 'date_id' in df.columns, "train_labels.csv 缺少 date_id"
    df['date_id'] = df['date_id'].astype('int32')
    df = df.replace(filler, np.nan).sort_values('date_id').reset_index(drop=True)
    return df

def read_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    assert {'target','lag','pair'} <= set(df.columns), "target_pairs.csv 列不齐"
    df['target'] = df['target'].astype(str)
    df['pair'] = df['pair'].astype(str)
    df['lag'] = df['lag'].astype(int)
    return df[['target','lag','pair']].copy()


# ============================================================
# B) 缺失值：停牌检测 + 常规缺失修复（无数据泄漏）
# ============================================================
def _is_discontinued_pattern(s: pd.Series, min_tail_ratio: float = 0.10) -> bool:
    if s.isna().all():
        return True
    last_valid = s.last_valid_index()
    if last_valid is None:
        return True
    pos = s.index.get_loc(last_valid)
    if pos < len(s) - 1:
        tail = s.iloc[pos+1:]
        if len(tail) >= max(3, int(len(s)*min_tail_ratio)) and tail.isna().all():
            return True
    return False

def detect_discontinued_columns(df: pd.DataFrame, threshold=0.8, exclude=("date_id",)) -> list[str]:
    cols=[]
    n=len(df)
    for c in df.columns:
        if c in exclude: continue
        miss = df[c].isna().mean()
        if miss>=threshold and _is_discontinued_pattern(df[c]):
            cols.append(c)
    return cols

def process_discontinued(df: pd.DataFrame, cols: list[str], strategy: str):
    if not cols: return df, []
    df = df.copy(); added=[]
    if strategy == "fill_last":
        for c in cols:
            if df[c].notna().any():
                last_val = df[c].ffill().iloc[-1]
                df[c] = df[c].ffill().fillna(last_val if pd.notna(last_val) else 0.0)
            else:
                df[c] = 0.0
    elif strategy == "indicator":
        for c in cols:
            ind = f"{c}_is_active"
            df[ind] = df[c].notna().astype(np.int8); added.append(ind)
            df[c] = df[c].ffill().fillna(0.0)
    elif strategy == "fill_zero":
        for c in cols: df[c] = df[c].fillna(0.0)
    elif strategy == "drop":
        df = df.drop(columns=cols)
    else:
        raise ValueError("unknown discontinued strategy")
    return df, added

def process_regular_missing(df: pd.DataFrame, strategy="forward_only", exclude=("date_id",)) -> pd.DataFrame:
    """
    修正版：避免数据泄漏，只使用前向填充
    """
    df = df.copy()
    for c in df.columns:
        if c in exclude: continue
        if df[c].isna().sum()==0: continue
        
        if strategy == "forward_only":
            # 只用前向填充，开头的NaN用0或历史均值填充
            s = df[c].ffill()
            # 对于序列开头的缺失值，用0填充（保守策略）
            s = s.fillna(0.0)
            df[c] = s
        elif strategy == "mean":
            m = df[c].mean()
            df[c] = df[c].fillna(0.0 if pd.isna(m) else m)
        elif strategy == "median":
            m = df[c].median()
            df[c] = df[c].fillna(0.0 if pd.isna(m) else m)
        else:
            raise ValueError(f"unknown regular strategy: {strategy}")
    return df

def preprocess_prices_with_missing_policy(train_df: pd.DataFrame) -> pd.DataFrame:
    cfg = CONFIG["MISSING"]
    discontinued = detect_discontinued_columns(
        train_df, threshold=cfg["discontinued_threshold"])
    df1, added = process_discontinued(train_df, discontinued, cfg["discontinued_strategy"])
    df2 = process_regular_missing(df1, cfg["regular_strategy"])
    print(f"[MISSING] discontinued={len(discontinued)} added_indicators={len(added)}  "
          f"NaNs: {train_df.isna().sum().sum()} -> {df2.isna().sum().sum()}")
    return df2


# ============================================================
# C) 解析 target_pairs（严格命中）
# ============================================================
def parse_pairs_strict(train_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows, miss = [], []
    for _, r in pairs_df.iterrows():
        tgt = str(r['target']).strip()
        pair_raw = "" if pd.isna(r['pair']) else str(r['pair']).strip()
        lag = int(r['lag'])
        if ' - ' in pair_raw:
            a, b = [x.strip() for x in pair_raw.split(' - ', 1)]
            ok = (a in train_df.columns) and (b in train_df.columns)
            if ok:
                rows.append({'target':tgt,'lag':lag,'a_col':a,'b_col':b,'is_spread':True})
            else:
                miss.append((tgt, pair_raw))
        else:
            a = pair_raw
            ok = (a in train_df.columns)
            if ok:
                rows.append({'target':tgt,'lag':lag,'a_col':a,'b_col':None,'is_spread':False})
            else:
                miss.append((tgt, pair_raw))
    if miss:
        print(f"[PAIR][WARN] {len(miss)} pairs not found in train cols, e.g. {miss[:3]}")
    out = pd.DataFrame(rows)
    assert out['target'].is_unique, "duplicate target in pairs!"
    return out


# ============================================================
# D) 特征工程（时间域 + 横截面 + 静态 PCA/聚类）
# ============================================================

def safe_log(x: np.ndarray, epsilon: float = None) -> np.ndarray:
    """安全的对数变换，避免log(0)或log(负数)"""
    if epsilon is None:
        epsilon = CONFIG["LOG_EPSILON"]
    return np.log(np.maximum(x, epsilon))

def _get_commodity_groups(columns):
    """识别合约月份并分组，将同一商品的合约归为一类"""
    # e.g., 'JPX_Gold_202308' -> 'JPX_Gold'
    groups = {}
    for col in columns:
        # 匹配 "交易所_品名_" 的模式
        match = re.match(r'([A-Z]+_[A-Za-z]+)_', col)
        if match:
            commodity = match.group(1)
            if commodity not in groups:
                groups[commodity] = []
            groups[commodity].append(col)
    return groups

def add_term_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算并添加期限结构特征 (斜率和Carry)"""
    print("Adding term structure (Slope/Carry) features...")
    df_out = df.copy()
    
    price_cols = [c for c in df.columns if c != 'date_id']
    commodity_groups = _get_commodity_groups(price_cols)
    
    for commodity, cols in commodity_groups.items():
        # 筛选出包含 'close' 或 'price' 的价格列
        price_cols_comm = [c for c in cols if 'close' in c.lower() or 'price' in c.lower()]
        if len(price_cols_comm) < 2:
            continue
            
        # 简单地将列按名称排序，通常能近似按到期日排序
        # 一个更稳健的方法是解析列名中的日期
        price_cols_comm = sorted(price_cols_comm)
        
        # 我们只计算近月和次近月之间的价差作为代表
        near_col = price_cols_comm[0]
        far_col = price_cols_comm[1]
        
        # 安全地取对数
        log_near = safe_log(df[near_col])
        log_far = safe_log(df[far_col])
        
        # 计算斜率 (Slope)
        # Backwardation (近>远) 时为正，Contango (近<远) 时为负
        df_out[f'slope_{commodity}'] = log_near - log_far
        
        # 计算Carry的代理指标 (Proxy)
        # 一个简化的Carry因子，数值越大，Backwardation程度越高，做多Carry收益越高
        df_out[f'carry_{commodity}'] = (log_near - log_far) / log_near.replace(0, 1e-8)
        
    return df_out

@dataclass
class FeatureEngineer:
    windows: Tuple[int,...] = field(default_factory=lambda: CONFIG["WINDOWS"])
    ewma_span: int = CONFIG["EWMA_SPAN"]
    corr_w: int = CONFIG["CORR_W"]
    pca_k: int = CONFIG["PCA_K"]
    kmeans_k: int = CONFIG["KMEANS_K"]
    seed: int = CONFIG["SEED"]

    # 训练得到的静态对象
    pca_components_: Optional[np.ndarray] = None
    kmeans_centers_: Optional[np.ndarray] = None
    target_order_: Optional[List[str]] = None
    target_cluster_: Optional[Dict[str,int]] = None
    
    # 新增：特征名称管理
    feature_names_: Set[str] = field(default_factory=set)

    def build_features_from_pairs(self, train_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        rows, date_idx = [], train_df['date_id'].values
        max_w = max(self.windows)
        self.feature_names_.clear()  # 重置特征名称集合
        
        for _, r in pairs_df[['target','a_col','b_col','is_spread']].iterrows():
            # 安全的对数处理
            a_raw = pd.to_numeric(train_df[r['a_col']], errors='coerce')
            a = safe_log(a_raw)
            
            if r['is_spread']:
                b_raw = pd.to_numeric(train_df[r['b_col']], errors='coerce')
                base = a - safe_log(b_raw)
            else:
                base = a
            
            # 修正：只使用前向填充，避免数据泄漏
            base = pd.Series(base).ffill()
            # 序列开头的NaN用0填充
            base = base.fillna(0.0)
            
            # 平滑处理：现在只使用指数移动平均 (EWM)
            z = base.ewm(span=self.ewma_span, adjust=False).mean()
            res = base - z  # 残差
            
            feat = pd.DataFrame({'date_id': date_idx, 'target': r['target'], 'z': z})
            
            # 核心特征
            feat['res'] = res  # KF残差（短期动量信号）
            feat['ret1'] = feat['z'].diff()
            self.feature_names_.update(['z', 'res', 'ret1'])
            
            # --- 新增代码：计算“关注度”因子 ---
            # 1. 构建交易量列的名称 (e.g., 'close_0' -> 'volume_0')
            volume_col = r['a_col'].replace('price', 'volume').replace('close', 'volume').replace('open', 'volume').replace('high', 'volume').replace('low', 'volume')
            if volume_col in train_df.columns:
                # 2. 计算原始关注度信号
                attention_raw = feat['ret1'].abs() * safe_log(train_df[volume_col])
                feat['attention'] = attention_raw
                
                # 3. 计算关注度的移动平均，使其更平滑
                feat['attention_ma20'] = attention_raw.rolling(20, min_periods=20).mean()
                
                # 4. 将新特征名称加入集合
                self.feature_names_.update(['attention', 'attention_ma20'])
            # --- 新增代码结束 ---
            
            # 技术指标特征
            for w in self.windows:
                feat[f'ma{w}'] = feat['z'].rolling(w, min_periods=w).mean()
                feat[f'vol{w}'] = feat['z'].diff().rolling(w, min_periods=w).std()
                feat[f'zscore_{w}'] = (feat['z'] - feat[f'ma{w}'])/(feat[f'vol{w}']+1e-6)
                self.feature_names_.update([f'ma{w}', f'vol{w}', f'zscore_{w}'])
            
            # RSI特征（相对强弱指标）
            delta = feat['z'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
            feat['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
            self.feature_names_.add('rsi')
            
            # 额外稳健项
            feat['absret'] = feat['ret1'].abs()
            feat['ema_vol10'] = feat['ret1'].ewm(span=10, adjust=False).std()
            feat['skew20'] = feat['ret1'].rolling(20, min_periods=20).skew()
            feat['kurt20'] = feat['ret1'].rolling(20, min_periods=20).kurt()
            feat['res_vol'] = res.rolling(10, min_periods=10).std()  # 残差波动率
            self.feature_names_.update(['absret', 'ema_vol10', 'skew20', 'kurt20', 'res_vol'])
            
            # ★★★ 新增：添加交互与组合特征 ★★★
            feat = self._add_interaction_features(feat)
            
            rows.append(feat)
        
        feats = pd.concat(rows, ignore_index=True)
        feats = feats.dropna(subset=[f'ma{max_w}', f'vol{max_w}']).reset_index(drop=True)
        return feats

    def _add_interaction_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        """
        ★★★ 新增方法 ★★★
        在基础特征之上，创建交互与组合特征。
        """
        # 1. 短期/长期动量比 (捕捉趋势变化)
        for short_w in [5, 10, 20]:
            for long_w in [60, 120, 252]:
                sw = f'zscore_{short_w}'
                lw = f'zscore_{long_w}'
                if sw in feat.columns and lw in feat.columns:
                    col_name = f'zscore_ratio_{short_w}_{long_w}'
                    feat[col_name] = feat[sw] / (feat[lw] + 1e-8)
                    self.feature_names_.add(col_name)

        # 2. 波动率调节收益 (类夏普比率)
        if 'ret1' in feat.columns and 'vol20' in feat.columns:
            feat['sharpe_proxy_20'] = feat['ret1'] / (feat['vol20'] + 1e-8)
            self.feature_names_.add('sharpe_proxy_20')

        # 3. 关注度调节波动率 (衡量放量波动)
        if 'vol20' in feat.columns and 'attention_ma20' in feat.columns:
            feat['attention_vol_20'] = feat['vol20'] * feat['attention_ma20']
            self.feature_names_.add('attention_vol_20')
        
        return feat

    def _apply_feature_lagging(self, feats: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        辅助方法：对指定特征应用滞后处理（按 target 的真实 L），以修正数据泄漏。
        """
        # 定义需要被滞后的特征列的前缀
        leaky_prefixes = ('swt_', 'hp_', 'stl_')
        cols_to_lag = [col for col in feats.columns if col.startswith(leaky_prefixes)]
        if not cols_to_lag:
            return feats

        # 构建 target -> lag 映射
        lag_map: Dict[str, int] = dict(zip(pairs_df['target'].astype(str), pairs_df['lag'].astype(int)))
        print(f"Applying per-target lag to {len(cols_to_lag)} leaky features (min={min(lag_map.values()) if lag_map else 0}, max={max(lag_map.values()) if lag_map else 0})…")

        # 按 target 分组，使用各自的 L 进行 shift（就地替换列，不改名）
        def _shift_group(g: pd.DataFrame) -> pd.DataFrame:
            L = int(lag_map.get(str(g.name), 0))
            if L <= 0:
                return g[cols_to_lag]
            return g[cols_to_lag].shift(L)

        shifted = feats.groupby('target', sort=False, group_keys=False).apply(_shift_group)
        feats.loc[:, cols_to_lag] = shifted.values
        return feats

    def add_cross_sectional_feats(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        byday = out.groupby('date_id', observed=True)
        
        # --- 截面 rank / z (保持不变) ---
        for col in ['z','ret1','vol20']:
            if col in out.columns:
                out[f'{col}_xrank'] = byday[col].rank(pct=True, method='average')
                self.feature_names_.add(f'{col}_xrank')
        
        for col in ['z','ret1','ma10']:
            if col in out.columns:
                out[f'{col}_xz'] = byday[col].transform(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-6))
                self.feature_names_.add(f'{col}_xz')
        
        # --- 新增代码：计算“市场宽度”因子 ---
        if 'ret1' in out.columns:
            # 计算当天上涨资产的比例
            market_breadth = byday['ret1'].transform(lambda s: (s > 0).mean())
            out['market_breadth'] = market_breadth
            self.feature_names_.add('market_breadth')
        # --- 新增代码结束 ---
        
        # --- 新增代码：计算“截面波动率”因子 ---
        for w in [60, 252]: # 选择60天和252天波动率进行排名
            vol_col = f'vol{w}'
            if vol_col in out.columns:
                # 对资产当天的历史波动率进行横向排名
                out[f'cs_vol_rank_{w}d'] = byday[vol_col].rank(pct=True, method='average')
                self.feature_names_.add(f'cs_vol_rank_{w}d')
        # --- 新增代码结束 ---

        # 市场均值 (保持不变)
        out['mkt_z'] = byday['z'].transform('mean')
        
        # 与市场滚动相关 (保持不变)
        out[f'corr_mkt_{self.corr_w}'] = out.groupby('target', observed=True, group_keys=False).apply(
            lambda g: g['z'].rolling(self.corr_w, min_periods=self.corr_w).corr(g['mkt_z'])
        )
        self.feature_names_.add(f'corr_mkt_{self.corr_w}')
        
        # 自相关 (保持不变)
        out['ret1_l1'] = out.groupby('target', observed=True)['ret1'].shift(1)
        self.feature_names_.add('ret1_l1')
        
        out['ac1_20'] = out.groupby('target', observed=True, group_keys=False).apply(
            lambda g: g['ret1'].rolling(20, min_periods=20).corr(g['ret1_l1'])
        )
        self.feature_names_.add('ac1_20')
        
        return out

    def _fit_pca_kmeans(self, feats_train: pd.DataFrame):
        feats_train = feats_train.drop_duplicates(subset=['date_id','target'], keep='last') \
                             .sort_values(['date_id','target'])
        mat = feats_train.pivot_table(index='date_id', columns='target',
                                      values='z_xz', aggfunc='last').fillna(0.0)
        targets = mat.columns.tolist()
        X = mat.values.astype('float32')
        
        if X.shape[1] < 3:
            self.pca_components_ = None
            self.kmeans_centers_ = None
            self.target_order_ = None
            self.target_cluster_ = None
            return
        
        pca = PCA(n_components=min(self.pca_k, X.shape[1]-1), random_state=self.seed)
        pca.fit(X)
        comps = pca.components_.astype('float32')
        loads = comps.T
        
        k = min(self.kmeans_k, max(2, X.shape[1]//5))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        clusters = km.fit_predict(loads)
        
        self.pca_components_ = comps
        self.kmeans_centers_ = km.cluster_centers_.astype('float32')
        self.target_order_ = targets
        self.target_cluster_ = {t:int(c) for t,c in zip(targets, clusters)}

    def _attach_static_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        if self.pca_components_ is None or self.target_order_ is None:
            return feats
        
        loads = self.pca_components_.T
        df_s = pd.DataFrame(loads, index=self.target_order_, 
                           columns=[f'pca_ld_{i+1}' for i in range(loads.shape[1])])
        
        # 记录PCA特征名称
        for i in range(loads.shape[1]):
            self.feature_names_.add(f'pca_ld_{i+1}')
        
        df_s.index.name = 'target'
        df_s = df_s.reset_index()
        
        if self.target_cluster_ is not None:
            df_s['cluster_id'] = df_s['target'].map(self.target_cluster_)
            for k in sorted(set(self.target_cluster_.values())):
                df_s[f'clu_{k}'] = (df_s['cluster_id']==k).astype('int8')
                self.feature_names_.add(f'clu_{k}')
        
        return feats.merge(df_s, on='target', how='left')

    def fit(self, train_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        feats_time = self.build_features_from_pairs(train_df, pairs_df)
        
        # --- 新增步骤：加入所有新特征 ---
        feats_time = add_swt_features(feats_time, series_col='z', wavelet='db4', level=4)
        feats_time = add_cs_momentum_features(feats_time, series_col='z')
        feats_time = add_hp_filter_features(feats_time, series_col='z')
        feats_time = add_stl_features(feats_time, series_col='z', period=21)

        new_feature_names = [
            'cs_momentum_120d', 'cs_momentum_252d',
            'hp_cycle', 'hp_trend',
            'stl_trend', 'stl_seasonal', 'stl_resid',
        ]
        for l in range(1, 5):
            new_feature_names.append(f'swt_z_cA{l}')
            new_feature_names.append(f'swt_z_cD{l}')
        self.feature_names_.update(new_feature_names)
        # --- 新增结束 ---
        # --- ★★★ 新增调用：对刚生成的潜在泄漏特征整体滞后 ★★★ ---
        feats_time = self._apply_feature_lagging(feats_time, pairs_df)
        # --- 修改结束 ---
        feats_full = self.add_cross_sectional_feats(feats_time)
        self._fit_pca_kmeans(feats_full)
        feats_full = self._attach_static_features(feats_full)
        feats_full = feats_full.drop_duplicates(subset=['date_id','target'], keep='last')
        return feats_full

    def transform(self, df_like_train: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        feats_time = self.build_features_from_pairs(df_like_train, pairs_df)
        
        # --- 同样在 transform 中加入完全相同的调用 ---
        feats_time = add_swt_features(feats_time, series_col='z', wavelet='db4', level=4)
        feats_time = add_cs_momentum_features(feats_time, series_col='z')
        feats_time = add_hp_filter_features(feats_time, series_col='z')
        feats_time = add_stl_features(feats_time, series_col='z', period=21)

        new_feature_names = [
            'cs_momentum_120d', 'cs_momentum_252d',
            'hp_cycle', 'hp_trend',
            'stl_trend', 'stl_seasonal', 'stl_resid',
        ]
        for l in range(1, 5):
            new_feature_names.append(f'swt_z_cA{l}')
            new_feature_names.append(f'swt_z_cD{l}')
        self.feature_names_.update(new_feature_names)
        # --- 新增结束 ---
        # --- ★★★ 新增调用：对刚生成的潜在泄漏特征整体滞后 ★★★ ---
        feats_time = self._apply_feature_lagging(feats_time, pairs_df)
        # --- 修改结束 ---
        feats_full = self.add_cross_sectional_feats(feats_time)
        feats_full = self._attach_static_features(feats_full)
        feats_full = feats_full.drop_duplicates(subset=['date_id','target'], keep='last')
        return feats_full
    
    def get_feature_columns(self) -> List[str]:
        """返回所有特征列名称"""
        return sorted(list(self.feature_names_))


# ============================================================
# E) lag 严格对齐：t 的特征 → 预测 t+lag 的标签
# ============================================================
def align_long_features_with_labels(feats_long: pd.DataFrame,
                                    labels_df: pd.DataFrame,
                                    pairs_df: pd.DataFrame) -> pd.DataFrame:
    lag_map = pairs_df[['target','lag']].copy()
    df = feats_long.merge(lag_map, on='target', how='left')
    assert df['lag'].notna().all(), "missing lag!"
    
    lab = labels_df.copy()
    lab_cols = [c for c in lab.columns if c!='date_id']
    long_lab = lab.melt(id_vars='date_id', value_vars=lab_cols,
                        var_name='target', value_name='y')
    long_lab = long_lab.merge(lag_map, on='target', how='inner')

    # 统一目标，不再对长周期进行平滑

    long_lab['date_id'] = long_lab['date_id'] - long_lab['lag']
    
    aligned = df.merge(long_lab[['date_id','target','y']], on=['date_id','target'], how='inner')
    aligned['y'] = aligned['y'].replace(CONFIG["LABELS_FILLER"], np.nan)
    aligned = aligned.dropna(subset=['y']).reset_index(drop=True)
    return aligned


# ============================================================
# F) 时间CV + 稳健特征筛选 + 评估
# ============================================================
def make_time_folds(dates: pd.Series, n_folds: int = 4, embargo: int | None = None):
    """
    时间折切分，加入 embargo 空窗，防止平滑/滤波跨边界泄漏。
    embargo: 以天计的隔离带，建议 >= 最长窗口（如 max(CONFIG["WINDOWS"]) 或季节周期）。
    """
    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    fs = max(1, n//n_folds)
    folds=[]
    emb = embargo if embargo is not None else max(CONFIG.get("WINDOWS", (1,)))
    for k in range(n_folds):
        va_start = k*fs
        va_end = (k+1)*fs if k < n_folds-1 else n
        va = set(uniq[va_start:va_end])
        if not va:
            continue
        # 训练集：去掉验证集两侧的 embargo 天
        tr_right_edge = max(0, va_start - emb)
        tr = set(uniq[:tr_right_edge])
        if len(tr)>0 and len(va)>0:
            folds.append((tr,va))
    return folds

def daily_spearman_sharpe(y, p, d) -> float:
    df = pd.DataFrame({'y':y,'p':p,'d':d}).replace([np.inf,-np.inf],np.nan).dropna()
    if df.empty: 
        return 0.0
    g = df.groupby('d').apply(lambda g_: g_['y'].rank().corr(g_['p'].rank(), method='spearman')) \
         .replace([np.inf,-np.inf],np.nan).dropna()
    return float(g.mean()/(g.std(ddof=0)+1e-12)) if len(g) else 0.0

def get_feature_cols(df: pd.DataFrame, feature_engineer: Optional[FeatureEngineer] = None) -> list:
    """改进版：使用FeatureEngineer管理的特征名称"""
    if feature_engineer and hasattr(feature_engineer, 'feature_names_'):
        # 使用FeatureEngineer记录的特征名称，并确保列存在
        cols = [c for c in feature_engineer.get_feature_columns() if c in df.columns]
        if cols:
            return cols
    
    # 回退到原始逻辑（保持向后兼容）
    blacklist={'date_id','target','y','lag','mkt_z','cluster_id'}
    cols=[]
    for c in df.columns:
        if c in blacklist: continue
        # --- 新增对 slope 和 carry 的识别 ---
        if c.startswith(('ma','vol','zscore_','pca_ld_','clu_','corr_mkt_', 'slope_', 'carry_')): 
        # --- 修改结束 ---
            cols.append(c)
            continue
        if c in {'z','ret1','absret','ema_vol10','skew20','kurt20','ret1_l1','ac1_20','res','res_vol','rsi'}: 
            cols.append(c)
            continue
        if c.endswith(('_xrank','_xz')): 
            cols.append(c)
            continue
    return sorted(cols)

def basic_feature_filters(df: pd.DataFrame, feature_cols: list, miss_thresh=0.25, corr_thresh=0.98):
    # 修复：检查是否有列
    if not feature_cols:
        print("[WARN] No feature columns to filter")
        return []
    
    X = df[feature_cols]
    keep = [c for c in feature_cols if X[c].isna().mean() <= miss_thresh]
    if not keep:
        print("[WARN] All features exceed missing threshold")
        return []
    
    X = X[keep]
    keep = [c for c in X.columns if X[c].std(ddof=0) > 1e-12]
    if not keep:
        print("[WARN] All features have zero variance")
        return []
    
    X = X[keep]
    
    # 相关性过滤
    if len(keep) > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop=set()
        for c in upper.columns:
            if c in drop: continue
            hits = set(upper.index[upper[c] > corr_thresh].tolist())
            drop |= hits
        keep = [c for c in X.columns if c not in drop]
    
    return keep


def make_daywise_relevance(y: np.ndarray, d: np.ndarray, top_q: float = 0.25) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    d = np.asarray(d)
    if y.shape[0] != d.shape[0]:
        raise ValueError("y 和 d 的长度必须一致。")
    if not (0.0 < top_q < 1.0):
        raise ValueError("top_q 必须在 (0,1) 区间内。")

    rel = np.zeros_like(y, dtype=np.int32)

    for day in np.unique(d):
        m = (d == day)
        if not np.any(m):
            continue

        # 仅用该日内的非 NaN y 参与分位数与比较
        m_valid = m & np.isfinite(y)
        y_day = y[m_valid]

        # 小样本或无方差 → 全部置 0
        if y_day.size < 3 or np.nanstd(y_day) == 0.0:
            rel[m] = 0
            continue

        thr = np.nanquantile(y_day, 1.0 - top_q)

        r_valid = (y_day >= thr).astype(np.int32)

        # 强制至少存在 1 个正类与 1 个负类
        if r_valid.sum() == 0:
            r_valid[np.nanargmax(y_day)] = 1
        if r_valid.sum() == r_valid.size:
            r_valid[np.nanargmin(y_day)] = 0

        # 写回：有效样本按计算结果；无效样本(该日内 y 为 NaN)置 0
        rel[m_valid] = r_valid
        rel[m & ~m_valid] = 0

    return rel

def _filter_invalid_groups_for_rank(X, y_cont, y_rel, d):
    """去掉只有1条样本 or 只有单一等级的日期组"""
    df = pd.DataFrame({'d': d, 'rel': y_rel})
    cnt = df.groupby('d').size()
    nuniq = df.groupby('d')['rel'].nunique()
    valid_days = cnt[(cnt >= 3) & (nuniq >= 2)].index
    m = np.isin(d, valid_days)
    return X[m], y_cont[m], y_rel[m], d[m]

def _train_ranker_with_fallback(X_tr_s, y_tr_s, d_tr_s, group_counts, seed):
    """统一的ranker训练逻辑，带多重回退机制"""
    lgb, _, _, _, _, _, _ = _maybe_import_training_libs()
    # 1) 尝试lambdarank + 二元标签
    y_rel = make_daywise_relevance(y_tr_s, d_tr_s, top_q=0.25)
    X1, y1, yrel1, d1 = _filter_invalid_groups_for_rank(X_tr_s, y_tr_s, y_rel, d_tr_s)
    _, gc1 = np.unique(d1, return_counts=True)
    
    if lgb is not None and len(d1) > 1000 and gc1.min() >= 3:
        try:
            ranker = lgb.LGBMRanker(
                objective='lambdarank',
                label_gain=[0,1],
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=10,
                min_data_in_leaf=10,
                min_sum_hessian_in_leaf=1e-6,
                feature_pre_filter=False,
                max_bin=511,
                colsample_bytree=0.9,
                subsample=0.8, subsample_freq=1,
                reg_alpha=0.2, reg_lambda=0.6,
                n_estimators=600,
                random_state=seed, n_jobs=-1,
                verbosity=-1
            )
            ranker.fit(X1, yrel1, group=gc1)
            imp = ranker.booster_.feature_importance(importance_type='gain')
            if np.sum(imp) > 0 and ranker.booster_.num_trees() > 1:
                return ranker, "lambdarank"
        except Exception as e:
            pass

    # 2) 兜底：rank_xendcg，连续非负标签
    if lgb is not None:
        try:
            y_nn = y_tr_s - pd.Series(y_tr_s).groupby(d_tr_s).transform('min').to_numpy()
            y_nn = y_nn + 1e-8
            ranker = lgb.LGBMRanker(
                objective='rank_xendcg',
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=10,
                min_data_in_leaf=10,
                min_sum_hessian_in_leaf=1e-6,
                feature_pre_filter=False,
                max_bin=511,
                colsample_bytree=0.9,
                subsample=0.8, subsample_freq=1,
                reg_alpha=0.2, reg_lambda=0.6,
                n_estimators=600,
                random_state=seed, n_jobs=-1,
                verbosity=-1
            )
            ranker.fit(X_tr_s, y_nn, group=group_counts)
            imp = ranker.booster_.feature_importance(importance_type='gain')
            if np.sum(imp) > 0 and ranker.booster_.num_trees() > 1:
                return ranker, "rank_xendcg"
        except Exception as e:
            pass

    # 3) 最后兜底：回归"日内秩"
    y_rank = pd.Series(y_tr_s).groupby(d_tr_s).rank(method='average').values
    if lgb is not None:
        reg = lgb.LGBMRegressor(
            objective='huber', huber_delta=1.0,
            learning_rate=0.05, num_leaves=63,
            min_child_samples=10, min_data_in_leaf=10,
            min_sum_hessian_in_leaf=1e-6,
            feature_pre_filter=False,
            max_bin=511,
            colsample_bytree=0.9,
            subsample=0.8, subsample_freq=1,
            reg_alpha=0.2, reg_lambda=0.6,
            n_estimators=600,
            random_state=seed, n_jobs=-1,
            verbosity=-1
        )
        reg.fit(X_tr_s, y_rank)
        return reg, "reg_rank"
    else:
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_tr_s, y_rank)
        return reg, "ridge_rank"

def select_features_by_cv(df: pd.DataFrame, feature_cols: list, n_keep=64, n_folds=3, seed=42):
    """修复版：增加空折和无效组过滤"""
    # 基础清洗
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=['y'])
    
    # 如果特征为空，返回空
    if not feature_cols:
        print("[WARN] No features to select")
        return [], pd.DataFrame()
    
    d = df['date_id'].values
    y = df['y'].values
    y_dir = (y > 0).astype(int)
    X = df[feature_cols].values
    
    # 检查数据量
    if len(df) < 100:
        print("[WARN] Too few samples for feature selection")
        # 返回保底特征
        fallback = ['z', 'ret1', 'ma5', 'vol5', 'ma10', 'vol10']
        selected = [f for f in fallback if f in feature_cols][:n_keep]
        return selected, pd.DataFrame()
    
    folds = make_time_folds(df['date_id'], n_folds=n_folds)
    
    l1_freq = pd.Series(0.0, index=feature_cols, dtype='float64')
    gbm_gain = pd.Series(0.0, index=feature_cols, dtype='float64')

    for tr_dates, va_dates in folds:
        tr_m = df['date_id'].isin(tr_dates).values
        va_m = df['date_id'].isin(va_dates).values
        if tr_m.sum()<500 or va_m.sum()<100: 
            continue
        
        X_tr, y_tr, d_tr = X[tr_m], y_dir[tr_m], d[tr_m]
        
        # 检查训练集是否为空
        if X_tr.shape[0] == 0:
            continue
        
        # 方向：L1-LogReg
        try:
            lr = make_pipeline(
                SimpleImputer(strategy='constant', fill_value=0.0),
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000, random_state=seed)
            )
            lr.fit(X_tr, y_tr)
            coef = lr.named_steps['logisticregression'].coef_.ravel()
            mask = (np.abs(coef) > 1e-8)
            l1_freq.loc[np.array(feature_cols)[mask]] += 1.0
        except Exception as e:
            print(f"[WARN] L1 LogReg failed in fold: {e}")
            continue

        # 幅度：LGBM Ranker - 修复：应用无效组过滤
        X_tr2 = X[tr_m]
        y_tr2 = df['y'].values[tr_m]
        d_tr2 = d_tr
        
        # 排序
        ord_tr = np.argsort(d_tr2, kind='mergesort')
        X_tr_s, y_tr_s, d_tr_s = X_tr2[ord_tr], y_tr2[ord_tr], d_tr2[ord_tr]
        
        # 过滤无效组
        y_rel = make_daywise_relevance(y_tr_s, d_tr_s, top_q=0.25)
        X_filtered, y_filtered, yrel_filtered, d_filtered = _filter_invalid_groups_for_rank(X_tr_s, y_tr_s, y_rel, d_tr_s)
        
        if len(d_filtered) < 100:  # 过滤后样本太少
            continue
        
        _, group_counts = np.unique(d_filtered, return_counts=True)
        
        try:
            model_rank, mode = _train_ranker_with_fallback(X_filtered, y_filtered, d_filtered, group_counts, seed)
            
            if hasattr(model_rank, "booster_"):
                gain = pd.Series(model_rank.booster_.feature_importance(importance_type='gain'),
                               index=feature_cols, dtype='float64')
            else:
                # 回归时用 feature_importances_
                gain = pd.Series(getattr(model_rank, "feature_importances_", np.zeros(len(feature_cols))),
                               index=feature_cols, dtype='float64')
            gbm_gain = gbm_gain.add(gain, fill_value=0.0)
        except Exception as e:
            print(f"[WARN] Ranker failed in fold: {e}")
            continue

    l1_norm = (l1_freq / max(1, len(folds))).astype('float64')
    gsum = gbm_gain.sum()
    gbm_norm = gbm_gain / (gsum if gsum>0 else 1.0)
    fused = 0.4 * l1_norm + 0.6 * gbm_norm
    rank = fused.sort_values(ascending=False)
    selected = rank.index[:n_keep].tolist()
    
    # 如果没有选中任何特征，返回保底特征
    if not selected:
        fallback = ['z', 'ret1', 'ma5', 'vol5', 'ma10', 'vol10', 'ma20', 'vol20']
        selected = [f for f in fallback if f in feature_cols][:n_keep]
    
    score_df = pd.DataFrame({
        'feature': rank.index,
        'score': rank.values,
        'l1_freq': l1_norm.loc[rank.index].values,
        'gbm_gain_norm': gbm_norm.loc[rank.index].values
    })
    return selected, score_df

# ============================================================
# G) 训练单个lag（LR 方向 + LGBM Rank 幅度 + 等权日内融合）
# ============================================================

def add_swt_features(df_long: pd.DataFrame, series_col: str, wavelet: str = 'db4', level: int = 4) -> pd.DataFrame:
    """
    对长格式DataFrame中的时间序列进行SWT分解，并添加特征。

    >>> 最终修正版：处理奇数长度 & 动态调整层数 <<<
    """
    print(f"Applying SWT (level={level}, wavelet='{wavelet}') on column '{series_col}'...")
    _, _, _, _, pywt, _, _ = _maybe_import_training_libs()

    output_dfs = []
    for _, group in df_long.groupby('target', sort=False):
        group = group.copy()
        series = group[series_col].values
        original_len = len(series)

        # 处理奇数长度：末尾补一个元素以满足SWT对偶数长度的要求
        is_odd = False
        if original_len % 2 != 0:
            is_odd = True
            series = np.pad(series, (0, 1), 'edge')

        if pywt is None:
            # 无 pywt 时，直接填充 0 特征
            for L in range(1, level + 1):
                group[f'swt_{series_col}_cA{L}'] = np.zeros(original_len)
                group[f'swt_{series_col}_cD{L}'] = np.zeros(original_len)
            output_dfs.append(group)
            continue

        # 根据当前序列长度计算能支持的最大层数
        max_level = pywt.swt_max_level(len(series))
        actual_level = min(level, max_level)

        # 如果连一层都无法分解，直接填0
        if actual_level < 1:
            for L in range(1, level + 1):
                group[f'swt_{series_col}_cA{L}'] = 0.0
                group[f'swt_{series_col}_cD{L}'] = 0.0
            output_dfs.append(group)
            continue

        try:
            coeffs = pywt.swt(series, wavelet=wavelet, level=actual_level, norm=True)

            for L in range(1, level + 1):
                if L <= actual_level:
                    idx = actual_level - L  # coeffs逆序返回
                    cA, cD = coeffs[idx]

                    if is_odd:
                        cA = cA[:-1]
                        cD = cD[:-1]

                    group[f'swt_{series_col}_cA{L}'] = cA
                    group[f'swt_{series_col}_cD{L}'] = cD
                else:
                    group[f'swt_{series_col}_cA{L}'] = np.zeros(original_len)
                    group[f'swt_{series_col}_cD{L}'] = np.zeros(original_len)
        except Exception as e:
            print(f"  [ERROR] SWT failed for a target group: {e}. Filling with 0.")
            for L in range(1, level + 1):
                group[f'swt_{series_col}_cA{L}'] = np.zeros(original_len)
                group[f'swt_{series_col}_cD{L}'] = np.zeros(original_len)

        output_dfs.append(group)

    if not output_dfs:
        return df_long

    df_with_swt = pd.concat(output_dfs, ignore_index=True)
    return df_with_swt

# === New Feature Functions ===
def add_cs_momentum_features(df_long: pd.DataFrame, series_col: str = 'z') -> pd.DataFrame:
    """计算并添加长期截面动量特征"""
    print("Adding long-term cross-sectional momentum features...")
    df_long = df_long.drop_duplicates(subset=['date_id','target'], keep='last') \
                     .sort_values(['date_id','target'])
    df_wide = df_long.pivot_table(index='date_id', columns='target',
                                  values=series_col, aggfunc='last')
    for period in [120, 252]:
        returns_wide = df_wide.diff(periods=period)
        ranks_wide = returns_wide.rank(axis=1, pct=True, method='average')
        feature_name = f'cs_momentum_{period}d'
        ranks_long = ranks_wide.melt(ignore_index=False, var_name='target', value_name=feature_name).reset_index()
        df_long = df_long.merge(ranks_long, on=['date_id', 'target'], how='left')
    return df_long

# removed: add_rolling_hurst_features (requires hurst)

def add_hp_filter_features(df_long: pd.DataFrame, series_col: str = 'z', strict_rolling: bool = True) -> pd.DataFrame:
    """使用HP滤波分解序列，并添加周期和趋势作为特征。
    strict_rolling=True 时，逐点仅使用到 t 为止的历史样本计算（防未来）。
    """
    print("Adding Hodrick-Prescott Filter features...")
    _, _, _, _, _, sm, _ = _maybe_import_training_libs()
    def apply_hp_filter(group):
        series = group[series_col]
        series_filled = series.ffill().fillna(0)
        if sm is None:
            group[f'hp_cycle'] = 0.0
            group[f'hp_trend'] = series_filled
            return group
        if not strict_rolling:
            if len(series_filled) > 2:
                cycle, trend = sm.tsa.filters.hpfilter(series_filled, lamb=1.6e7)
                group[f'hp_cycle'] = cycle
                group[f'hp_trend'] = trend
            else:
                group[f'hp_cycle'] = 0.0
                group[f'hp_trend'] = series_filled
            return group

        # 严格滚动计算（仅用到 t 的历史）
        n = len(series_filled)
        if n == 0:
            group[f'hp_cycle'] = 0.0
            group[f'hp_trend'] = series_filled
            return group
        trend_vals = np.zeros(n, dtype=float)
        cycle_vals = np.zeros(n, dtype=float)
        for i in range(n):
            sub = series_filled.iloc[:i+1]
            if len(sub) > 2:
                try:
                    c, t = sm.tsa.filters.hpfilter(sub, lamb=1.6e7)
                    cycle_vals[i] = float(c.iloc[-1])
                    trend_vals[i] = float(t.iloc[-1])
                except Exception:
                    cycle_vals[i] = 0.0
                    trend_vals[i] = float(sub.iloc[-1])
            else:
                cycle_vals[i] = 0.0
                trend_vals[i] = float(sub.iloc[-1])
        group[f'hp_cycle'] = cycle_vals
        group[f'hp_trend'] = trend_vals
        return group
    df_with_hp = df_long.groupby('target', sort=False, group_keys=False).apply(apply_hp_filter)
    return df_with_hp

def add_stl_features(df_long: pd.DataFrame, series_col: str = 'z', period: int = 21, strict_rolling: bool = True) -> pd.DataFrame:
    """
    使用STL分解时间序列，并添加趋势、季节性和残差作为特征。

    :param df_long: 包含 'target' 和 'date_id' 的长格式DataFrame
    :param series_col: 需要被分解的序列所在的列名 (例如 'z')
    :param period: 季节性周期。对于日度数据，常用的有5(周)、21(月)
    :return: 带有新STL特征的DataFrame
    """
    print(f"Adding STL features (period={period}) on column '{series_col}'...")
    _, _, _, _, _, _, STL = _maybe_import_training_libs()
    output_dfs = []
    for _, group in df_long.groupby('target', sort=False):
        group = group.copy()
        series = group[series_col]
        series_filled = series.ffill().fillna(0)
        if not strict_rolling:
            if len(series_filled) < period * 2 or STL is None:
                if len(series_filled) < period * 2:
                    print(f"  [WARN] Series for a target is too short for STL (len={len(series_filled)}, period={period}). Filling with 0.")
                group['stl_trend'] = series_filled
                group['stl_seasonal'] = 0.0
                group['stl_resid'] = 0.0
            else:
                try:
                    stl = STL(series_filled, period=period)
                    result = stl.fit()
                    group['stl_trend'] = result.trend
                    group['stl_seasonal'] = result.seasonal
                    group['stl_resid'] = result.resid
                except Exception as e:
                    print(f"  [ERROR] STL failed for a target group: {e}. Filling with 0.")
                    group['stl_trend'] = series_filled
                    group['stl_seasonal'] = 0.0
                    group['stl_resid'] = 0.0
        else:
            # 严格滚动：逐点仅使用到 t 的历史进行 STL
            n = len(series_filled)
            trend_vals = np.zeros(n, dtype=float)
            seas_vals = np.zeros(n, dtype=float)
            resid_vals = np.zeros(n, dtype=float)
            for i in range(n):
                sub = series_filled.iloc[:i+1]
                if STL is None or len(sub) < period * 2:
                    trend_vals[i] = float(sub.iloc[-1])
                    seas_vals[i] = 0.0
                    resid_vals[i] = 0.0
                else:
                    try:
                        stl = STL(sub, period=period)
                        result = stl.fit()
                        trend_vals[i] = float(result.trend.iloc[-1])
                        seas_vals[i] = float(result.seasonal.iloc[-1])
                        resid_vals[i] = float(result.resid.iloc[-1])
                    except Exception:
                        trend_vals[i] = float(sub.iloc[-1])
                        seas_vals[i] = 0.0
                        resid_vals[i] = 0.0
            group['stl_trend'] = trend_vals
            group['stl_seasonal'] = seas_vals
            group['stl_resid'] = resid_vals
        output_dfs.append(group)
    if not output_dfs:
        return df_long
    df_with_stl = pd.concat(output_dfs, ignore_index=True)
    return df_with_stl
# 已移除 add_emd_features 与 add_ssa_features
# 已移除混合损失相关自定义函数
def winsorize_by_day(df_: pd.DataFrame, col='y', p=0.005):
    def _w(g):
        lo, hi = g[col].quantile([p, 1-p])
        g[col] = g[col].clip(lo, hi)
        return g
    return df_.groupby('date_id', observed=True, group_keys=False).apply(_w)

def make_dataset_for_lag(L, aligned_all, pairs_df, fe):
    """为指定lag准备数据集和特征"""
    tgts = pairs_df.loc[pairs_df['lag'] == L, 'target']
    ds = aligned_all[aligned_all['target'].isin(tgts)].copy()
    if len(ds) == 0:
        return None, []
    # 取特征列（尊重FeatureEngineer记录的列）
    feat_cols = get_feature_cols(ds, fe)
    feat_cols = basic_feature_filters(ds, feat_cols, miss_thresh=0.25, corr_thresh=0.98)
    if not feat_cols:
        fallback = ['z','ret1','ma5','vol5','ma10','vol10','ma20','vol20']
        feat_cols = [f for f in fallback if f in ds.columns]
    return ds, feat_cols

def objective_factory(L, aligned_all, pairs_df, fe, seed=CONFIG["SEED"]):
    """创建针对特定lag的Optuna目标函数（按需导入 optuna，类型注解可缺省）"""
    def _objective(trial):
        # 1) 超参搜索空间（先聚焦 LGBM + LR + 特征数量）
        params = {
            # Logistic Regression
            'logreg_C': trial.suggest_float('logreg_C', 1e-3, 1.0, log=True),

            # LightGBM ranker (修改搜索空间)
            'lgbm_learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.15),
            'lgbm_num_leaves': trial.suggest_int('lgbm_num_leaves', 31, 127),
            'lgbm_max_depth': trial.suggest_int('lgbm_max_depth', 5, 10),  # 给出更深的探索空间
            
            # --- 核心修改：允许模型学习更细的规则 ---
            'lgbm_min_data_in_leaf': trial.suggest_int('lgbm_min_data_in_leaf', 10, 50),  # 将下限从32大幅降低到10
            
            'lgbm_feature_fraction': trial.suggest_float('lgbm_feature_fraction', 0.6, 0.9),
            'lgbm_bagging_fraction': trial.suggest_float('lgbm_bagging_fraction', 0.6, 0.9),
            'lgbm_reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0.0, 1.0),  # 减小正则化范围
            'lgbm_reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0.0, 2.0),  # 减小正则化范围
            
            'lgbm_n_estimators': 1500,  # 可以固定为一个更大的值，完全依赖早停

            # （可选）特征选择的保留数量
            'fs_keep': trial.suggest_int('fs_keep', 24, 96, step=8),
        }

        ds, feat_cols = make_dataset_for_lag(L, aligned_all, pairs_df, fe)
        if ds is None or len(feat_cols) == 0:
            # 不可训练，给个很差的值
            return -1e9

        # 2) 先用你已有的CV特征筛选，保留 fs_keep
        selected, _ = select_features_by_cv(
            ds, feat_cols, 
            n_keep=params['fs_keep'], 
            n_folds=CONFIG["N_FOLDS"], 
            seed=seed
        )
        if not selected:
            return -1e9

        # 3) 训练并拿 Sharpe（best_fusion）
        metrics, _, _ = train_one_lag_enhanced(
            aligned_df=ds, feature_cols=selected, lag=L, 
            n_folds=CONFIG["N_FOLDS"], seed=seed,
            model_params=params
        )
        # 目标：最大化 best_sharpe（已按日IC Sharpe算）
        return metrics['best_sharpe']
    return _objective

def train_one_lag_enhanced(aligned_df: pd.DataFrame, 
                           feature_cols: list, 
                           lag: int, 
                           n_folds: int = 3, 
                           seed: int = 42,
                           model_params: dict | None = None):
    """
    ★★★ Stacking增强版 ★★★
    """
    if model_params is None: model_params = {}
    lgb, xgb, CatBoostRegressor, _, _, _, _ = _maybe_import_training_libs()
    df = aligned_df.copy()
    df = winsorize_by_day(df, col='y', p=0.005)

    if not feature_cols:
        print("[ERROR] No features for training")
        return {'sharpe_p':0.0,'sharpe_r':0.0,'sharpe_mult':0.0,'sharpe_stack':0.0,'best_fusion':'mult_rank','best_sharpe':0.0}, \
               {'feature_cols':[],'lr_models':[],'rank_models':[],'fusion_params':{}}, \
               {'p':np.array([]),'r':np.array([]),'y':np.array([]),'d':np.array([])}

    y, y_dir, d, X = df['y'].values, (df['y'].values > 0).astype(int), df['date_id'].values, df[feature_cols].values
    folds = make_time_folds(df['date_id'], n_folds=n_folds)

    # ★★★ 1. 初始化OOF数组 ★★★
    oof_p = np.full(len(df), np.nan)
    oof_r_lgbm = np.full(len(df), np.nan)
    oof_r_xgb = np.full(len(df), np.nan)
    oof_r_cat = np.full(len(df), np.nan)

    models_lr, models_lgbm, models_xgb, models_cat = [], [], [], []

    for fold_idx, (tr_dates, va_dates) in enumerate(folds):
        tr_mask = df['date_id'].isin(tr_dates).values
        va_mask = df['date_id'].isin(va_dates).values
        if tr_mask.sum()<300 or va_mask.sum()<50:
            continue

        X_tr = X[tr_mask]; X_va = X[va_mask]
        if X_tr.shape[0] == 0:
            continue
        y_tr_dir = y_dir[tr_mask]
        y_tr_cont = y[tr_mask]
        d_tr = d[tr_mask]

        # 方向模型：RF
        X_tr_fold, X_va_fold, y_tr_dir_fold, y_va_dir_fold = train_test_split(
            X_tr, y_tr_dir, test_size=0.2, shuffle=False
        )
        try:
            rf_pipeline = make_pipeline(
                SimpleImputer(strategy='constant', fill_value=0.0),
                StandardScaler(),
                RandomForestClassifier(
                    n_estimators=200, max_depth=5, min_samples_leaf=50,
                    class_weight='balanced', random_state=seed, n_jobs=-1
                )
            )
            rf_pipeline.fit(X_tr_fold, y_tr_dir_fold)
            oof_p[va_mask] = rf_pipeline.predict_proba(X_va)[:, 1]
            models_lr.append(rf_pipeline)
        except Exception as e:
            print(f"[WARN] RandomForest training failed in fold {fold_idx}: {e}")
            continue

        # 排序模型数据
        ord_tr = np.argsort(d_tr, kind='mergesort')
        X_tr_s = X_tr[ord_tr]; y_tr_s = y_tr_cont[ord_tr]; d_tr_s = d_tr[ord_tr]
        _, group_counts = np.unique(d_tr_s, return_counts=True)

        # LGBM
        if lgb is not None:
            try:
                ranker_params = dict(
                    objective='lambdarank', feature_fraction=model_params.get('lgbm_feature_fraction', 0.75),
                    bagging_fraction=model_params.get('lgbm_bagging_fraction', 0.75), bagging_freq=1,
                    min_data_in_leaf=model_params.get('lgbm_min_data_in_leaf', 64), min_gain_to_split=0.0,
                    num_leaves=model_params.get('lgbm_num_leaves', 63), max_depth=model_params.get('lgbm_max_depth', -1),
                    learning_rate=model_params.get('lgbm_learning_rate', 0.05), n_estimators=model_params.get('lgbm_n_estimators', 400),
                    reg_alpha=model_params.get('lgbm_reg_alpha', 1.0), reg_lambda=model_params.get('lgbm_reg_lambda', 3.0),
                    force_col_wise=True, random_state=seed + fold_idx, n_jobs=-1, verbosity=1
                )
                ranker = lgb.LGBMRanker(**ranker_params)
                y_rel = pd.cut(pd.Series(y_tr_s).groupby(d_tr_s).rank(pct=True), bins=[0,0.2,0.4,0.6,0.8,1.0], labels=[0,1,2,3,4], include_lowest=True).astype(int).values
                (X_tr_s_fold, X_va_s_fold, y_rel_fold, y_va_rel_fold, d_tr_s_fold, d_va_s_fold) = train_test_split(X_tr_s, y_rel, d_tr_s, test_size=0.2, shuffle=False)
                _, group_counts_train = np.unique(d_tr_s_fold, return_counts=True)
                _, group_counts_val = np.unique(d_va_s_fold, return_counts=True)
                ranker.fit(
                    X_tr_s_fold, y_rel_fold, group=group_counts_train,
                    eval_set=[(X_va_s_fold, y_va_rel_fold)], eval_group=[group_counts_val], eval_metric='ndcg',
                    callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=50, verbose=True)]
                )
                oof_r_lgbm[va_mask] = ranker.predict(X_va)
                models_lgbm.append(ranker)
            except Exception as e:
                print(f"[WARN] Fold {fold_idx} LGBM training failed: {e}")

        # XGBoost
        if xgb is not None:
            try:
                dtrain_xgb = xgb.DMatrix(X_tr_s_fold, label=y_rel_fold); dtrain_xgb.set_group(group_counts_train)
                deval_xgb = xgb.DMatrix(X_va_s_fold, label=y_va_rel_fold); deval_xgb.set_group(group_counts_val)
                xgb_params = {
                    'objective': 'rank:pairwise', 'eval_metric': 'ndcg',
                    'eta': model_params.get('xgb_eta', 0.05), 'max_depth': model_params.get('xgb_max_depth', 5),
                    'subsample': model_params.get('xgb_subsample', 0.8), 'colsample_bytree': model_params.get('xgb_colsample_bytree', 0.8),
                    'seed': seed + fold_idx
                }
                xgb_ranker = xgb.train(params=xgb_params, dtrain=dtrain_xgb, num_boost_round=model_params.get('xgb_n_estimators', 400),
                                       evals=[(dtrain_xgb,'train'),(deval_xgb,'eval')], verbose_eval=100, early_stopping_rounds=50)
                dtest_xgb_va = xgb.DMatrix(X_va)
                oof_r_xgb[va_mask] = xgb_ranker.predict(dtest_xgb_va)
                models_xgb.append(xgb_ranker)
            except Exception as e:
                print(f"[WARN] Fold {fold_idx} XGBoost training failed: {e}")

        # CatBoost
        if CatBoostRegressor is not None:
            try:
                y_rank_cat = pd.Series(y_tr_s).groupby(d_tr_s).rank(pct=True).values
                (X_tr_s_fold_cat, X_va_s_fold_cat, y_rank_cat_fold, y_va_rank_cat_fold) = train_test_split(X_tr_s, y_rank_cat, test_size=0.2, shuffle=False)
                cat_ranker = CatBoostRegressor(
                    iterations=model_params.get('cat_iterations', 1000), learning_rate=model_params.get('cat_learning_rate', 0.05),
                    depth=model_params.get('cat_depth', 6), l2_leaf_reg=model_params.get('cat_l2_leaf_reg', 3.0),
                    loss_function='RMSE', random_seed=seed + fold_idx, verbose=100
                )
                cat_ranker.fit(X_tr_s_fold_cat, y_rank_cat_fold, eval_set=[(X_va_s_fold_cat, y_va_rank_cat_fold)], early_stopping_rounds=50, use_best_model=True)
                oof_r_cat[va_mask] = cat_ranker.predict(X_va)
                models_cat.append(cat_ranker)
            except Exception as e:
                print(f"[WARN] Fold {fold_idx} CatBoost training failed: {e}")

    # 融合策略（Stacking）
    valid = ~np.isnan(oof_p) & ~np.isnan(oof_r_lgbm) & ~np.isnan(oof_r_xgb) & ~np.isnan(oof_r_cat)
    if valid.sum()==0:
        return {'sharpe_p':0.0,'sharpe_r':0.0,'sharpe_mult':0.0,'sharpe_stack':0.0,'best_fusion':'mult_rank','best_sharpe':0.0}, \
               {'feature_cols':feature_cols,'lr_models':models_lr,'rank_models':[],'fusion_params':{}}, \
               {'p':oof_p,'r_lgbm':oof_r_lgbm,'y':y,'d':d}

    df_eval = pd.DataFrame({
        'y': y[valid], 'p': oof_p[valid], 'd': d[valid],
        'r_lgbm': oof_r_lgbm[valid], 'r_xgb': oof_r_xgb[valid], 'r_cat': oof_r_cat[valid]
    })

    # Stacking元模型
    X_meta_rank = df_eval[['r_lgbm','r_xgb','r_cat']].values
    y_meta_rank = df_eval['y'].values
    ranker_stacker_final = Ridge(alpha=1.0, random_state=seed)
    ranker_stacker_final.fit(X_meta_rank, y_meta_rank)
    df_eval['r_stacking'] = cross_val_predict(ranker_stacker_final, X_meta_rank, y_meta_rank, cv=min(3, len(np.unique(df_eval['d']))) )

    df_eval['p_rank'] = df_eval.groupby('d')['p'].rank(pct=True)
    df_eval['r_stacking_rank'] = df_eval.groupby('d')['r_stacking'].rank(pct=True)
    df_eval['fuse_mult_rank'] = df_eval['p_rank'] * df_eval['r_stacking_rank']

    sharpe_p = daily_spearman_sharpe(df_eval['y'], df_eval['p'], df_eval['d'])
    sharpe_r = daily_spearman_sharpe(df_eval['y'], df_eval['r_stacking'], df_eval['d'])
    sharpe_mult = daily_spearman_sharpe(df_eval['y'], df_eval['fuse_mult_rank'], df_eval['d'])

    # 不确定性模型
    print("    Training uncertainty model...")
    uncertainty_model = None
    X_uncertainty_train = X[valid]
    y_uncertainty_train = np.abs(oof_p[valid] - y_dir[valid])
    if len(X_uncertainty_train) > 100:
        if lgb is not None:
            try:
                uncertainty_model = lgb.LGBMRegressor(
                    objective='regression_l1', metric='mae', learning_rate=0.05,
                    n_estimators=200, num_leaves=10, max_depth=3, min_child_samples=100, min_data_in_leaf=100,
                    subsample=0.7, colsample_bytree=0.7, reg_alpha=3.0, reg_lambda=5.0,
                    random_state=seed, n_jobs=-1, verbosity=-1
                )
                uncertainty_model.fit(X_uncertainty_train, y_uncertainty_train)
                print("    Uncertainty model trained successfully.")
            except Exception as e:
                print(f"[WARN] Uncertainty model training failed: {e}")
                uncertainty_model = None
    else:
        print("    Not enough OOF samples to train uncertainty model.")
    
    metrics = {
        'sharpe_p': sharpe_p,
        'sharpe_r': sharpe_r,
        'sharpe_mult': sharpe_mult,
        'sharpe_stack': sharpe_mult,
        'best_fusion': 'mult_rank',
        'best_sharpe': sharpe_mult
    }
    
    artifacts = {
        'feature_cols': feature_cols,
        'lr_models': models_lr,
        'lgbm_models': models_lgbm,
        'xgb_models': models_xgb,
        'cat_models': models_cat,
        'ranker_stacker': ranker_stacker_final,
        'fusion_params': {'strategy': 'mult_rank'},
        'uncertainty_model': uncertainty_model
    }
    
    oof = {'p': oof_p, 'r_lgbm': oof_r_lgbm, 'y': y, 'd': d}
    
    return metrics, artifacts, oof
    
    # 原混合损失分支已前移为早返回，这里删除重复代码


# ============================================================
# H) 一键离线流程（读取→缺失→pairs→特征→对齐→筛选→训练）
# ============================================================
def run_offline_train():
    base = CONFIG["INPUT_DIR"]
    train_df  = read_train(os.path.join(base, "train.csv"))
    labels_df = read_labels(os.path.join(base, "train_labels.csv"), CONFIG["LABELS_FILLER"])
    pairs_raw = read_pairs(os.path.join(base, "target_pairs.csv"))
    print(f"[INFO] raw shapes  train {train_df.shape}  labels {labels_df.shape}  pairs {len(pairs_raw)}")

    # 缺失处理
    train_df = preprocess_prices_with_missing_policy(train_df)

    # --- 新增步骤：在原始价格数据上生成期限结构特征 ---
    train_df = add_term_structure_features(train_df)
    # --- 新增结束 ---

    # pairs 严格解析
    pairs_df = parse_pairs_strict(train_df, pairs_raw)
    lag_counts = pairs_df['lag'].value_counts().sort_index().to_dict()
    print(f"[PAIR] usable targets: {len(pairs_df)}  lag_counts={lag_counts}")

    # 特征工程（训练拟合）
    fe = FeatureEngineer(
        windows=CONFIG["WINDOWS"],
        ewma_span=CONFIG["EWMA_SPAN"],
        corr_w=CONFIG["CORR_W"],
        pca_k=CONFIG["PCA_K"], kmeans_k=CONFIG["KMEANS_K"],
        seed=CONFIG["SEED"]
    )
    feats_long = fe.fit(train_df, pairs_df)
    print(f"[FEATS] long: {feats_long.shape}")

    # 严格 lag 对齐
    aligned_all = align_long_features_with_labels(feats_long, labels_df, pairs_df)
    aligned_all = aligned_all.replace([np.inf, -np.inf], np.nan)
    print(f"[ALIGN] aligned_all: {aligned_all.shape}")

    # 分 lag 训练
    summary = {}
    artifacts_by_lag = {}
    feature_scores_by_lag = {}
    use_enhanced = True  # 使用增强版训练流程
    
    for L in (1, 2, 3, 4):
        tgts = pairs_df.loc[pairs_df['lag'] == L, 'target']
        ds = aligned_all[aligned_all['target'].isin(tgts)].copy()
        if len(ds) == 0:
            print(f"[lag={L}] empty, skip")
            continue
        print(f"[lag={L}] samples: {ds.shape}")

        feat_cols = get_feature_cols(ds, fe)  # 传入 fe 以使用管理的特征名称
        feat_cols = basic_feature_filters(ds, feat_cols, miss_thresh=0.25, corr_thresh=0.98)
        
        if not feat_cols:
            print(f"[lag={L}] No features after filtering, using fallback")
            fallback = ['z', 'ret1', 'ma5', 'vol5', 'ma10', 'vol10', 'ma20', 'vol20']
            feat_cols = [f for f in fallback if f in ds.columns]
        
        selected, score_df = select_features_by_cv(
            ds, feat_cols, n_keep=CONFIG["FS_KEEP"], 
            n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"]
        )
        feature_scores_by_lag[L] = score_df
        print(f"[lag={L}] selected {len(selected)} feats → {selected[:6]}")

        # 使用增强版训练（分离 LR 和 LGBM，多种融合策略）
        metrics, artifacts, oof = train_one_lag_enhanced(
            ds, selected, L, n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"]
        )
        summary[L] = metrics
        artifacts_by_lag[L] = {
            "artifacts": artifacts,
            "oof": oof,
            "selected_features": selected
        }
        if "sharpe_r" in metrics:
            print(
                f"[METRIC] L{L}: LR={metrics['sharpe_p']:.3f}  "
                f"Rank={metrics['sharpe_r']:.3f}  MultRank={metrics['sharpe_mult']:.3f}  "
                f"Stack={metrics['sharpe_stack']:.3f}  Best({metrics['best_fusion']})={metrics['best_sharpe']:.3f}"
            )
        else:
            val = metrics.get("sharpe", np.nan)
            print(f"[METRIC] L{L}: sharpe={val:.3f}")

    # 修复版：正确打印汇总（检查是否为增强版）
    print("\n[Done] 四个 lag 的线下指标：")
    for L in (1, 2, 3, 4):
        if L not in summary:
            print(f"  L{L}: (no data)")
            continue
        m = summary[L]
        if "sharpe_r" in m:  # 增强版
            print(
                f"  L{L}: p={m['sharpe_p']:.3f}, r={m['sharpe_r']:.3f}, "
                f"mult={m['sharpe_mult']:.3f}, stack={m['sharpe_stack']:.3f}, "
                f"best({m['best_fusion']})={m['best_sharpe']:.3f}"
            )
        else:
            # 兼容基础版指标结构
            val = m.get("sharpe", np.nan)
            print(f"  L{L}: sharpe={val:.3f}")

    return summary, artifacts_by_lag, feature_scores_by_lag



# ============================================================
# I) 使用固定的最优参数进行最终训练
# ============================================================
def run_final_training_with_best_params(best_params_dict: dict):
    """
    ★★★ 新函数 ★★★
    使用固定的最优参数字典来执行完整的最终模型训练。
    """
    base = CONFIG["INPUT_DIR"]
    train_df  = read_train(os.path.join(base, "train.csv"))
    labels_df = read_labels(os.path.join(base, "train_labels.csv"), CONFIG["LABELS_FILLER"])
    pairs_raw = read_pairs(os.path.join(base, "target_pairs.csv"))
    print(f"[INFO] raw shapes  train {train_df.shape}  labels {labels_df.shape}  pairs {len(pairs_raw)}")

    train_df = preprocess_prices_with_missing_policy(train_df)
    train_df = add_term_structure_features(train_df)
    pairs_df = parse_pairs_strict(train_df, pairs_raw)
    
    fe = FeatureEngineer(
        windows=CONFIG["WINDOWS"],
        ewma_span=CONFIG["EWMA_SPAN"],
        corr_w=CONFIG["CORR_W"],
        pca_k=CONFIG["PCA_K"], kmeans_k=CONFIG["KMEANS_K"],
        seed=CONFIG["SEED"]
    )
    feats_long = fe.fit(train_df, pairs_df)
    aligned_all = align_long_features_with_labels(feats_long, labels_df, pairs_df)
    aligned_all = aligned_all.replace([np.inf, -np.inf], np.nan)
    print(f"[ALIGN] aligned_all: {aligned_all.shape}")

    summary = {}
    artifacts_by_lag = {}
    feature_scores_by_lag = {}
    
    for L in (1, 2, 3, 4):
        params = best_params_dict.get(L, {})
        if not params:
            print(f"[lag={L}] 未在参数字典中找到配置，跳过")
            continue

        ds, all_feat_cols = make_dataset_for_lag(L, aligned_all, pairs_df, fe)
        if ds is None:
            continue

        fs_keep_optimal = params.get('fs_keep', CONFIG["FS_KEEP"])
        selected, score_df = select_features_by_cv(
            ds, all_feat_cols, n_keep=fs_keep_optimal,
            n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"]
        )
        feature_scores_by_lag[L] = score_df
        print(f"[lag={L}] selected {len(selected)} feats (using optimal fs_keep={fs_keep_optimal})")

        metrics, artifacts, oof = train_one_lag_enhanced(
            ds, selected, L, n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"],
            model_params=params
        )
        summary[L] = metrics
        artifacts_by_lag[L] = {
            "artifacts": artifacts, "oof": oof, "selected_features": selected
        }
        print(f"[METRIC] L{L} (Final Model): Best({metrics['best_fusion']})={metrics['best_sharpe']:.3f}")

    print("\n[Done] 四个 lag 的最终模型线下指标：")
    for L in (1, 2, 3, 4):
        if L not in summary:
            print(f"  L{L}: (no data)")
            continue
        m = summary[L]
        if "sharpe_r" in m:
            print(
                f"  L{L}: p={m['sharpe_p']:.3f}, r={m['sharpe_r']:.3f}, "
                f"mult={m['sharpe_mult']:.3f}, stack={m['sharpe_stack']:.3f}, "
                f"best({m['best_fusion']})={m['best_sharpe']:.3f}"
            )
        else:
            val = m.get("sharpe", np.nan)
            print(f"  L{L}: sharpe={val:.3f}")

    return summary, artifacts_by_lag, feature_scores_by_lag
# ============ Train & Save Bundle ============
import os, pickle
from copy import deepcopy

def fit_models_and_save(input_dir: str,
                        fs_keep: int = 64,
                        n_folds: int = 3,
                        seed: int = 42,
                        out_path: str = "/kaggle/working/model_bundle.pkl"):
    # 1) 读取数据
    train_df  = read_train(os.path.join(input_dir, "train.csv"))
    labels_df = read_labels(os.path.join(input_dir, "train_labels.csv"), CONFIG["LABELS_FILLER"])
    pairs_raw = read_pairs(os.path.join(input_dir, "target_pairs.csv"))
    print(f"[BUNDLE] train={train_df.shape} labels={labels_df.shape} pairs={len(pairs_raw)}")

    # 2) 缺失策略
    train_df = preprocess_prices_with_missing_policy(train_df)

    # 3) 解析 pairs
    pairs_df = parse_pairs_strict(train_df, pairs_raw)
    lag_counts = pairs_df['lag'].value_counts().sort_index().to_dict()
    print(f"[BUNDLE] pairs usable={len(pairs_df)}  lag_counts={lag_counts}")

    # 4) 特征工程（fit）
    fe = FeatureEngineer(
        windows=CONFIG["WINDOWS"],
        ewma_span=CONFIG["EWMA_SPAN"],
        corr_w=CONFIG["CORR_W"],
        pca_k=CONFIG["PCA_K"], kmeans_k=CONFIG["KMEANS_K"],
        seed=CONFIG["SEED"]
    )
    feats_long = fe.fit(train_df, pairs_df)
    print(f"[BUNDLE] feats_long={feats_long.shape}")

    # 5) 对齐标签
    aligned_all = align_long_features_with_labels(feats_long, labels_df, pairs_df)
    aligned_all = aligned_all.replace([np.inf, -np.inf], np.nan)
    print(f"[BUNDLE] aligned_all={aligned_all.shape}")

    # 6) 分 lag 进行特征筛选 + 训练
    per_lag = {}
    for L in (1,2,3,4):
        tgts = pairs_df.loc[pairs_df['lag']==L, 'target']
        ds = aligned_all[aligned_all['target'].isin(tgts)].copy()
        if ds.empty:
            print(f"[BUNDLE][lag={L}] empty, skip")
            continue

        feat_cols = get_feature_cols(ds, fe)
        feat_cols = basic_feature_filters(ds, feat_cols, miss_thresh=0.25, corr_thresh=0.98)

        selected, score_df = select_features_by_cv(
            ds, feat_cols, n_keep=fs_keep, n_folds=n_folds, seed=seed
        )
        print(f"[BUNDLE][lag={L}] keep={len(selected)}  head={selected[:6]}")

        # 训练（时间CV）→ 保存模型与校准器
        metrics, artifacts, oof = train_one_lag_enhanced(
            ds, selected, L, n_folds=n_folds, seed=seed
        )
        print(f"[BUNDLE][lag={L}] p={metrics['sharpe_p']:.3f}  "
              f"r={metrics['sharpe_r']:.3f}  mult={metrics['sharpe_mult']:.3f}  "
              f"stack={metrics['sharpe_stack']:.3f}  best({metrics['best_fusion']})={metrics['best_sharpe']:.3f}")

        per_lag[L] = {
            "features": selected,
            "lr_models": artifacts["lr_models"],
            # 替换 'rank_models'
            "lgbm_models": artifacts.get("lgbm_models", []),
            "xgb_models": artifacts.get("xgb_models", []),
            "cat_models": artifacts.get("cat_models", []),
            "fusion_params": artifacts.get("fusion_params", {}),
            "uncertainty_model": artifacts.get("uncertainty_model", None),
            "iso": None,  # 在线 predict_one_batch 里会判断 None，不会再取 iso.predict
        }

    # 7) 打包：为了在线端高效复用
    target_to_lag = dict(zip(pairs_df['target'], pairs_df['lag']))
    bundle = {
        "config": deepcopy(CONFIG),
        "pairs_df": pairs_df[['target','lag','a_col','b_col','is_spread']].copy(),
        "target_to_lag": target_to_lag,
        "feature_engineer": fe,   # 已 fit 的 PCA/聚类等
        "per_lag": per_lag
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[BUNDLE] saved → {out_path}")
    return bundle


# ============ Online Inference Server ============
import os, pickle
import pandas as pd
import polars as pl
import numpy as np

import kaggle_evaluation.mitsui_inference_server

NUM_TARGET_COLUMNS = 424

class OnlinePredictor:
    def __init__(self, input_dir: str, bundle_path: str = "/kaggle/working/model_bundle.pkl"):
        self.input_dir = input_dir
        self.bundle_path = bundle_path
        self.bundle = None
        self.prices_hist = None   # 累积（train + 流式 test）
        self._loaded = False

    def _lazy_load_or_fit(self):
        if self._loaded:
            return
        # 优先加载已训练好的 bundle；如果没有，现场训练一遍（首次 predict 允许>1min）
        if os.path.exists(self.bundle_path):
            try:
                with open(self.bundle_path, "rb") as f:
                    self.bundle = pickle.load(f)
                print("[ONLINE] load bundle from working")
            except Exception as e:
                print(f"[ONLINE][WARN] failed to load bundle ({e}); will fit on first call")
                self.bundle = None
        else:
            print("[ONLINE] bundle not found; fitting now (first call can be slow)…")
            self.bundle = fit_models_and_save(self.input_dir, 
                                              fs_keep=CONFIG["FS_KEEP"], 
                                              n_folds=CONFIG["N_FOLDS"], 
                                              seed=CONFIG["SEED"], 
                                              out_path=self.bundle_path)

        # 载入公开 train 以便构造滚动特征
        self.prices_hist = read_train(os.path.join(self.input_dir, "train.csv"))
        self.prices_hist = preprocess_prices_with_missing_policy(self.prices_hist)
        self._loaded = True

    def _ensure_columns(self, preds: pd.DataFrame) -> pd.DataFrame:
        # 保证按 target_0..target_423 全列返回（缺的补 0）
        cols = [f"target_{i}" for i in range(NUM_TARGET_COLUMNS)]
        out = pd.DataFrame({c: 0.0 for c in cols}, index=[0])
        for c in preds.columns:
            if c in out.columns:
                out[c] = preds[c].values
        return out

    def predict_one_batch(self, test_row_pl: pl.DataFrame) -> pd.DataFrame:
        """
        输入：当前时间步的 Polars 行（含 date_id 与所有价格列）。
        输出：1×424 的 pandas.DataFrame（target_0...target_423）。
        """
        self._lazy_load_or_fit()

        # ---- 1) 并入历史并按同策略补缺 ----
        test_row_pd = test_row_pl.to_pandas()
        assert len(test_row_pd) == 1, "每批应为单行"
        for c in test_row_pd.columns:
            if c != "date_id":
               test_row_pd[c] = pd.to_numeric(test_row_pd[c], errors='coerce').astype('float32')
        test_row_pd['date_id'] = test_row_pd['date_id'].astype('int32')

        self.prices_hist = pd.concat([self.prices_hist, test_row_pd], ignore_index=True)
        
        # 限制历史窗口大小，确保特征计算有足够数据但控制内存使用
        MAX_HISTORY_DAYS = 300
        if len(self.prices_hist) > MAX_HISTORY_DAYS:
            self.prices_hist = self.prices_hist.iloc[-MAX_HISTORY_DAYS:].reset_index(drop=True)
        
        self.prices_hist = preprocess_prices_with_missing_policy(self.prices_hist)

        # ---- 2) 特征（线上 transform；PCA/聚类已在离线 fit）----
        fe = self.bundle["feature_engineer"]
        pairs_df = self.bundle["pairs_df"]
        feats_long = fe.transform(self.prices_hist, pairs_df)

        cur_date = int(test_row_pd['date_id'].iloc[0])
        feats_cur = feats_long[feats_long['date_id'] == cur_date].copy()
        feats_cur = feats_cur.drop_duplicates(subset=['target'], keep='last')
        if feats_cur.empty:
           # 热身期不够或极端情况：全 0
           return pd.DataFrame({f"target_{i}": [0.0] for i in range(NUM_TARGET_COLUMNS)})

        per_lag = self.bundle["per_lag"]
        target_to_lag = self.bundle["target_to_lag"]

        preds = {}

        # 预先为每个 lag 构建 "位置索引对齐" 的矩阵，并容错缺列
        feats_by_lag = {}
        for L, info in per_lag.items():
            sel = info["features"]
            dfL = feats_cur.copy().reset_index(drop=True)       # ★关键：重建 0..N-1 索引
            # 用 reindex 保证所有 sel 列都在（缺的列为 NaN）
            X_df = dfL.reindex(columns=sel)
            # inf→nan→0
            X = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float, copy=False)
            # 保存一个 "target -> 位置" 的映射，避免每次搜索
            pos_map = {t: i for i, t in enumerate(dfL['target'].values)}
            feats_by_lag[L] = (dfL, X, pos_map, info)

        # 遍历当前日所有 target
        for tgt in feats_cur['target'].unique():
            L = int(target_to_lag.get(tgt, 1))
            if L not in feats_by_lag:
                preds[tgt] = 0.0
                continue
            dfL, X, pos_map, info = feats_by_lag[L]

            r = pos_map.get(tgt, None)
            if r is None:
                 preds[tgt] = 0.0
                 continue
            if r < 0 or r >= X.shape[0]:
                 # 保险起见，再防一层
                 preds[tgt] = 0.0
                 continue

            x1 = X[r:r+1, :]          # ★保证 1×F，不会出现 0×F

            # p：LR 均值
            if info["lr_models"]:
                p_list = [m.predict_proba(x1)[:, 1] for m in info["lr_models"]]
                p_hat = float(np.mean(p_list))
            else:
                p_hat = 0.5

            # ★★★ 使用stacking元模型融合排序评分 ★★★
            lgb, xgb, CatBoostRegressor, _, _, _, _ = _maybe_import_training_libs()
            lgbm_pred, xgb_pred, cat_pred = 0.0, 0.0, 0.0
            
            if info.get("lgbm_models"):
                try:
                    lgbm_preds = [m.predict(x1) for m in info["lgbm_models"]]
                    lgbm_pred = float(np.mean(lgbm_preds))
                except Exception:
                    lgbm_pred = 0.0

            if xgb is not None and info.get("xgb_models"):
                try:
                    dtest_xgb = xgb.DMatrix(x1)
                    xgb_preds = [m.predict(dtest_xgb) for m in info["xgb_models"]]
                    xgb_pred = float(np.mean(xgb_preds))
                except Exception:
                    xgb_pred = 0.0

            if info.get("cat_models"):
                try:
                    cat_preds = [m.predict(x1) for m in info["cat_models"]]
                    cat_pred = float(np.mean(cat_preds))
                except Exception:
                    cat_pred = 0.0

            ranker_stacker = info.get("ranker_stacker", None)
            if ranker_stacker:
                meta_features = np.array([[lgbm_pred, xgb_pred, cat_pred]])
                m_hat = float(ranker_stacker.predict(meta_features)[0])
            else:
                m_hat = float(np.mean([lgbm_pred, xgb_pred, cat_pred]))

            core = (2.0 * p_hat - 1.0) * m_hat
            
            # --- 新增代码：使用不确定性模型输出 sigma 并调整 core ---
            sigma_hat = 0.0 # 默认不确定性为0
            
            # 从 bundle 中获取不确定性模型
            uncertainty_model = info.get("uncertainty_model", None)
            
            if uncertainty_model is not None:
                try:
                    # 预测当前样本的不确定性 (误差的期望值)
                    sigma_hat = uncertainty_model.predict(x1)[0]
                except:
                    pass # 如果预测失败，则保持 sigma_hat 为0

            # 将不确定性 sigma (0-1之间) 转化为置信度 confidence (1-0之间)
            # 使用 np.clip 确保 sigma 在合理范围内
            confidence = 1.0 - np.clip(sigma_hat, 0, 1)
            
            # 用置信度来调整(缩放)最终的信号强度
            core = core * confidence
            # --- 新增代码结束 ---
            
            if info["iso"] is not None:
                core = float(info["iso"].predict([core])[0])

            preds[tgt] = core

        # ---- 4) 组装 1×424，缺列补 0 ----
        out = pd.DataFrame({t: [v] for t, v in preds.items()})
        need_cols = [f"target_{i}" for i in range(NUM_TARGET_COLUMNS)]
        out = out.reindex(columns=need_cols, fill_value=0.0)
        return out

    def predict_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        输入：pd.DataFrame（可能包含多行），每行是一批时点。
        输出：n×424 的 pandas.DataFrame，列名 target_0...target_423。
        """
        self._lazy_load_or_fit()
        if batch is None or len(batch) == 0:
            return pd.DataFrame({f"target_{i}": [] for i in range(NUM_TARGET_COLUMNS)})

        results = []
        # 逐行转换为 Polars 单行并复用现有在线路径
        for _, row in batch.iterrows():
            row_df = pd.DataFrame([row.to_dict()])
            # 保持列类型一致
            if 'date_id' in row_df.columns:
                row_df['date_id'] = pd.to_numeric(row_df['date_id'], errors='coerce').astype('int32')
            for c in row_df.columns:
                if c != 'date_id':
                    row_df[c] = pd.to_numeric(row_df[c], errors='coerce').astype('float32')
            test_row_pl = pl.from_pandas(row_df)
            res = self.predict_one_batch(test_row_pl)
            # ensure 424 columns
            res = self._ensure_columns(res)
            results.append(res)

        out = pd.concat(results, ignore_index=True)
        # 最终再确保一次列完整性与顺序
        need_cols = [f"target_{i}" for i in range(NUM_TARGET_COLUMNS)]
        out = out.reindex(columns=need_cols, fill_value=0.0)
        return out


# Kaggle Evaluation 必需接口（重命名，避免与线上 batch predict 冲突）
_predictor_singleton = OnlinePredictor(
    input_dir="/kaggle/input/mitsui-commodity-prediction-challenge",
    bundle_path="/kaggle/working/model_bundle.pkl"
)

def predict_kaggle(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame | pd.DataFrame:
    """
    返回 1×424。内部会懒加载/训练并缓存模型与价格历史。
    """
    return _predictor_singleton.predict_one_batch(test)

# 单例式在线预测器（竞赛评测端 batch 调用版本）
_online = OnlinePredictor(CONFIG["INPUT_DIR"])

def predict(batch: pd.DataFrame) -> pd.DataFrame:
    # 确保懒加载（含 bundle 载入/或一次性离线拟合）
    _online._lazy_load_or_fit()
    # 批量预测，返回 424 列
    out = _online.predict_batch(batch)
    # 校验列数，不足补齐
    assert out.shape[1] == NUM_TARGET_COLUMNS, "预测列数不为 424"
    return out


# ========== Optuna 超参数优化代码 ==========
def run_optuna_optimization(
    aligned_all,
    pairs_df,
    fe,
    lags_to_optimize=(1, 2, 3, 4),
    n_trials=40,
    seed=CONFIG["SEED"],
    lags=None,
):
    """
    运行Optuna超参数优化，为每个lag找到最优参数。

    支持两种入参方式：
    - 旧版：lags_to_optimize
    - 新版：lags（等价于 lags_to_optimize，优先级更高）

    Args:
        aligned_all: 对齐后的数据
        pairs_df: 交易对数据
        fe: 特征工程器
        lags_to_optimize: 要优化的lag列表（旧参数名）
        n_trials: 每个lag的试验次数
        seed: 随机种子
        lags: 要优化的lag列表（新参数名，若提供则优先生效）

    Returns:
        tuple: (best_params_by_lag, best_feats_by_lag, summary_optuna, artifacts_optuna)
    """
    best_params_by_lag = {}
    best_feats_by_lag = {}
    lags_to_use = lags if lags is not None else lags_to_optimize
    
    # 按需导入 optuna 及其组件，缺失则直接返回空结果
    _, _, _, optuna, _, _, _ = _maybe_import_training_libs()
    if optuna is None:
        print("[OPTUNA] optuna not available, skip optimization.")
        return {}, {}, {}, {}

    for L in lags_to_use:
        print(f"\n=== Optuna for lag={L} ===")
        # 准备数据（若某个lag没有样本会被跳过）
        ds, feat_cols = make_dataset_for_lag(L, aligned_all, pairs_df, fe)
        if ds is None:
            print(f"[lag={L}] empty, skip")
            continue

        objective = objective_factory(L, aligned_all, pairs_df, fe, seed=seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
        best_params_by_lag[L] = study.best_params

        # 用最优 fs_keep 再算一次最终的 selected
        ds_final, feat_cols_final = make_dataset_for_lag(L, aligned_all, pairs_df, fe)
        selected, _ = select_features_by_cv(
            ds_final, feat_cols_final, 
            n_keep=study.best_params.get('fs_keep', CONFIG["FS_KEEP"]),
            n_folds=CONFIG["N_FOLDS"], seed=seed
        )
        best_feats_by_lag[L] = selected

    # 用最优参数做一次"正式训练/汇总"
    summary_optuna = {}
    artifacts_optuna = {}
    for L, params in best_params_by_lag.items():
        ds, _ = make_dataset_for_lag(L, aligned_all, pairs_df, fe)
        selected = best_feats_by_lag[L]
        metrics, arts, oof = train_one_lag_enhanced(
            aligned_df=ds, feature_cols=selected, lag=L, 
            n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"],
            model_params=params
        )
        summary_optuna[L] = metrics
        artifacts_optuna[L] = arts

    return best_params_by_lag, best_feats_by_lag, summary_optuna, artifacts_optuna


# ========== 数据质量诊断函数 ==========
def check_group_quality(df_valid, group_col='g', label_col='y'):
    """
    检查group规模和label多样性
    
    Args:
        df_valid: 验证数据集
        group_col: group列名
        label_col: 标签列名
    
    Returns:
        dict: 包含各种统计信息的字典
    """
    print("=== Group质量检查 ===")
    
    # y: relevance, g: group id（通常是"日期"或"日期×lag"）
    grp = df_valid.groupby(group_col).agg(
        n=('y','size'),
        pos=('y', lambda s: (s>0).sum()),
        uniq=('y','nunique')
    )
    
    print("Group统计信息:")
    print(grp.describe())
    print(f"单样本组比例: {(grp['n']==1).mean():.4f}")
    print(f"单一标签组比例: {(grp['uniq']==1).mean():.4f}")
    print(f"仅单个正样本的组比例: {((grp['pos']==1) & (grp['n']>=2)).mean():.4f}")
    
    return {
        'group_stats': grp.describe(),
        'single_sample_ratio': (grp['n']==1).mean(),
        'single_label_ratio': (grp['uniq']==1).mean(),
        'single_positive_ratio': ((grp['pos']==1) & (grp['n']>=2)).mean()
    }

def check_feature_variance(X_valid, date_col='date', threshold=1e-12):
    """
    检查特征方差和常量列
    
    Args:
        X_valid: 特征数据
        date_col: 日期列名
        threshold: 方差阈值
    
    Returns:
        dict: 包含常量列统计信息的字典
    """
    print("\n=== 特征方差检查 ===")
    
    # 按日检查常量列
    bad_cols = []
    for d, sub in X_valid.groupby(date_col):
        zero_var = sub.loc[:, sub.columns].std(axis=0) < threshold
        bad_cols.extend(list(sub.columns[zero_var]))
    
    from collections import Counter
    bad_cols_counter = Counter(bad_cols)
    print("最常出现的常量列 (前20):")
    print(bad_cols_counter.most_common(20))
    
    # 全局检查常量列
    global_zero_var = X_valid.std(axis=0) < threshold
    global_bad_cols = X_valid.columns[global_zero_var].tolist()
    print(f"\n全局常量列数量: {len(global_bad_cols)}")
    print(f"全局常量列比例: {len(global_bad_cols) / len(X_valid.columns):.4f}")
    
    return {
        'daily_bad_cols': bad_cols_counter,
        'global_bad_cols': global_bad_cols,
        'global_bad_ratio': len(global_bad_cols) / len(X_valid.columns)
    }

def comprehensive_data_diagnosis(df, feature_cols, group_col='date_id', label_col='y'):
    """
    综合数据质量诊断
    
    Args:
        df: 完整数据集
        feature_cols: 特征列列表
        group_col: group列名
        label_col: 标签列名
    
    Returns:
        dict: 包含所有诊断结果的字典
    """
    print("=" * 60)
    print("开始综合数据质量诊断...")
    print("=" * 60)
    
    # 准备数据
    df_valid = df[df[label_col].notna()].copy()
    X_valid = df_valid[feature_cols].copy()
    
    # 1. Group质量检查
    group_quality = check_group_quality(df_valid, group_col, label_col)
    
    # 2. 特征方差检查
    feature_quality = check_feature_variance(X_valid, group_col)
    
    # 3. 基本统计信息
    print(f"\n=== 基本统计信息 ===")
    print(f"总样本数: {len(df_valid)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"Group数量: {df_valid[group_col].nunique()}")
    print(f"标签分布: {df_valid[label_col].value_counts().to_dict()}")
    
    # 4. 缺失值检查
    missing_stats = X_valid.isnull().sum()
    high_missing_cols = missing_stats[missing_stats > len(X_valid) * 0.5]
    print(f"\n高缺失率特征 (>50%): {len(high_missing_cols)}")
    if len(high_missing_cols) > 0:
        print("高缺失率特征列表:")
        print(high_missing_cols.head(10))
    
    return {
        'group_quality': group_quality,
        'feature_quality': feature_quality,
        'basic_stats': {
            'total_samples': len(df_valid),
            'feature_count': len(feature_cols),
            'group_count': df_valid[group_col].nunique(),
            'label_distribution': df_valid[label_col].value_counts().to_dict()
        },
        'missing_stats': {
            'high_missing_count': len(high_missing_cols),
            'high_missing_cols': high_missing_cols.head(10).to_dict()
        }
    }

# 示例使用代码（可在Notebook中手动执行）

# ========== 日志记录功能 ==========
import sys
from datetime import datetime

class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def setup_logging():
    """设置日志记录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # 重定向stdout到TeeOutput
    tee = TeeOutput(log_filename)
    sys.stdout = tee
    
    print(f"日志记录已启动，输出将保存到: {log_filename}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return tee, log_filename

def restore_logging(tee):
    """恢复标准输出"""
    sys.stdout = tee.terminal
    tee.close()
    print(f"日志记录已结束，文件已保存")

def ensure_pipeline_ready(CONFIG):
    """
    Notebook 友好：强制保证 aligned_all / pairs_df / fe 都已就绪。
    若全局已有则直接返回，否则按最小流程构建并提升到全局。
    """
    if 'aligned_all' in globals() and 'pairs_df' in globals() and 'fe' in globals():
        return globals()['aligned_all'], globals()['pairs_df'], globals()['fe']

    # 1) 读入原始数据
    train_df  = read_train(os.path.join(CONFIG["INPUT_DIR"], "train.csv"))
    labels_df = read_labels(os.path.join(CONFIG["INPUT_DIR"], "train_labels.csv"), CONFIG["LABELS_FILLER"])
    pairs_raw = read_pairs(os.path.join(CONFIG["INPUT_DIR"], "target_pairs.csv"))

    # 2) 缺失处理（只前向填充，避免泄漏）
    train_df = preprocess_prices_with_missing_policy(train_df)

    # 3) 严格解析 pairs（校验列名）
    pairs_ok = parse_pairs_strict(train_df, pairs_raw)

    # 4) 特征工程（fit）
    fe_local = FeatureEngineer()
    feats_long = fe_local.fit(train_df, pairs_ok)

    # 5) 标签严格对齐（t 的特征 → 预测 t+lag 的 y）
    aligned = align_long_features_with_labels(feats_long, labels_df, pairs_ok)

    # 6) 提升到全局，便于后续 cell 直接使用
    globals()['aligned_all'] = aligned
    globals()['pairs_df']    = pairs_ok
    globals()['fe']          = fe_local

    print("[READY] aligned_all / pairs_df / fe 已构建完毕：", aligned.shape)
    return aligned, pairs_ok, fe_local

def nb_prepare_data_and_features():
    """Notebook 友好：准备 aligned_all, pairs_df, fe 三个核心对象。"""
    print("--- [NB] Preparing base data and features ---")
    base = CONFIG["INPUT_DIR"]
    train_df  = read_train(os.path.join(base, "train.csv"))
    labels_df = read_labels(os.path.join(base, "train_labels.csv"), CONFIG["LABELS_FILLER"])
    pairs_raw = read_pairs(os.path.join(base, "target_pairs.csv"))

    train_df = preprocess_prices_with_missing_policy(train_df)
    train_df = add_term_structure_features(train_df)
    pairs_df = parse_pairs_strict(train_df, pairs_raw)

    fe = FeatureEngineer(
        windows=CONFIG["WINDOWS"],
        ewma_span=CONFIG["EWMA_SPAN"],
        corr_w=CONFIG["CORR_W"],
        pca_k=CONFIG["PCA_K"], kmeans_k=CONFIG["KMEANS_K"],
        seed=CONFIG["SEED"]
    )
    feats_long = fe.fit(train_df, pairs_df)
    aligned_all = align_long_features_with_labels(feats_long, labels_df, pairs_df)
    aligned_all = aligned_all.replace([np.inf, -np.inf], np.nan)

    print("aligned_all shape:", aligned_all.shape)
    print("sample:", aligned_all[["date_id","target","y"]].head())
    print("--- [NB] Data preparation complete! ---")
    return aligned_all, pairs_df, fe


def nb_run_optuna_one_lag(aligned_all, pairs_df, fe, lag=1, n_trials=50):
    """Notebook 友好：仅针对一个 lag 运行 Optuna 优化，并打印结果。"""
    print("\n--- [NB] Running Optuna (single lag) ---")
    best_params_by_lag, best_feats_by_lag, summary_optuna, artifacts_optuna = run_optuna_optimization(
        aligned_all=aligned_all,
        pairs_df=pairs_df,
        fe=fe,
        lags_to_optimize=(lag,),
        n_trials=n_trials
    )
    print("\n=== Optuna Optimization Results ===")
    for L, metrics in summary_optuna.items():
        print(f"Lag {L}: Best Sharpe = {metrics['best_sharpe']:.4f}")
        print(f"  Best fusion: {metrics['best_fusion']}")
        print(f"  Selected features: {len(best_feats_by_lag[L])}")
        print(f"  Best params: {best_params_by_lag[L]}")
        print("-" * 50)
    return best_params_by_lag, best_feats_by_lag, summary_optuna, artifacts_optuna


inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict_kaggle)

import os

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
