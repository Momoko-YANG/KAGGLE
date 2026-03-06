# ============================================================
# Mitsui Clean Baseline — DataIO → Missing → Pairs → Features
#   → Lag Align → Feature Selection (time-CV) → Train/Eval
# ============================================================

from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterable, Optional, Dict, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer

import lightgbm as lgb


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
        "regular_strategy": "smart_fill",     # 'smart_fill' | 'mean' | 'median'
    },

    # 特征工程
    "WINDOWS": (5, 10, 20, 60),
    "EWMA_SPAN": 10,
    "USE_KF": False,              # 想开 KF = True（本版自带简易 KF）
    "KF_Q": 1e-4, "KF_R": 1e-2,
    "CORR_W": 20,                 # 与“截面均值”的滚动相关窗口
    "PCA_K": 4, "KMEANS_K": 12,   # 静态 per-target 特征（可关）

    # 特征筛选 & 训练
    "N_FOLDS": 3,
    "FS_KEEP": 64,                # 每个 lag 保留的特征数（方向/幅度融合打分）
}


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
    # 强制转字符串，防止 method 被传递
    df['target'] = df['target'].astype(str)
    df['pair'] = df['pair'].astype(str)
    df['lag'] = df['lag'].astype(int)
    return df[['target','lag','pair']].copy()


# ============================================================
# B) 缺失值：停牌检测 + 常规缺失修复
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

def process_regular_missing(df: pd.DataFrame, strategy="smart_fill", exclude=("date_id",)) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in exclude: continue
        if df[c].isna().sum()==0: continue
        if strategy == "smart_fill":
            # 开头 bfill，其他 ffill，避免未来信息
            s = df[c].ffill().bfill().fillna(0.0)
            df[c] = s
        elif strategy == "mean":
            m = df[c].mean(); df[c] = df[c].fillna(0.0 if pd.isna(m) else m)
        elif strategy == "median":
            m = df[c].median(); df[c] = df[c].fillna(0.0 if pd.isna(m) else m)
        else:
            raise ValueError("unknown regular strategy")
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
        pair_raw = "" if pd.isna(r['pair']) else str(r['pair']).strip()  # 务必调用 .strip()
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
# D) KF（可选）+ 特征工程（时间域 + 横截面 + 静态 PCA/聚类）
# ============================================================
def _kf_smooth_1d(y: np.ndarray, q: float, r: float, x0: Optional[float]=None, p0: float=100.0) -> np.ndarray:
    # Local Level KF（仅 Filter）
    x = float(y[~np.isnan(y)][0]) if x0 is None else float(x0)
    p = float(p0)
    out = np.empty_like(y, dtype='float32')
    for i, obs in enumerate(y):
        x_pred = x
        p_pred = p + q
        if not np.isnan(obs):
            s = p_pred + r
            k = 0.0 if s <= 0 else (p_pred / s)
            x = x_pred + k * (obs - x_pred)
            p = (1.0 - k) * p_pred
        else:
            x, p = x_pred, p_pred
        out[i] = x
    return out

@dataclass
class FeatureEngineer:
    windows: Tuple[int,...] = field(default_factory=lambda: CONFIG["WINDOWS"])
    ewma_span: int = CONFIG["EWMA_SPAN"]
    use_kf: bool = CONFIG["USE_KF"]
    kf_Q: float = CONFIG["KF_Q"]
    kf_R: float = CONFIG["KF_R"]
    corr_w: int = CONFIG["CORR_W"]
    pca_k: int = CONFIG["PCA_K"]
    kmeans_k: int = CONFIG["KMEANS_K"]
    seed: int = CONFIG["SEED"]

    # 训练得到的静态对象
    pca_components_: Optional[np.ndarray] = None
    kmeans_centers_: Optional[np.ndarray] = None
    target_order_: Optional[List[str]] = None
    target_cluster_: Optional[Dict[str,int]] = None

    def build_features_from_pairs(self, train_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        rows, date_idx = [], train_df['date_id'].values
        max_w = max(self.windows)
        for _, r in pairs_df[['target','a_col','b_col','is_spread']].iterrows():
            a = np.log(pd.to_numeric(train_df[r['a_col']], errors='coerce'))
            base = a if not r['is_spread'] else (a - np.log(pd.to_numeric(train_df[r['b_col']], errors='coerce')))
            # 首端 bfill + 中段 ffill
            base = base.copy()
            fv = base.first_valid_index()
            if fv is not None and fv > 0:
                base.iloc[:fv] = base.iloc[fv]
            base = base.ffill()
            # 平滑
            if self.use_kf:
                z = pd.Series(_kf_smooth_1d(base.values.astype('float32'), self.kf_Q, self.kf_R), index=base.index)
            else:
                z = base.ewm(span=self.ewma_span, adjust=False).mean()
            feat = pd.DataFrame({'date_id': date_idx, 'target': r['target'], 'z': z})
            feat['ret1'] = feat['z'].diff()
            for w in self.windows:
                feat[f'ma{w}']  = feat['z'].rolling(w, min_periods=w).mean()
                feat[f'vol{w}'] = feat['z'].diff().rolling(w, min_periods=w).std()
                feat[f'zscore_{w}'] = (feat['z'] - feat[f'ma{w}'])/(feat[f'vol{w}']+1e-6)
            # 额外稳健项
            feat['absret']    = feat['ret1'].abs()
            feat['ema_vol10'] = feat['ret1'].ewm(span=10, adjust=False).std()
            feat['skew20']    = feat['ret1'].rolling(20, min_periods=20).skew()
            feat['kurt20']    = feat['ret1'].rolling(20, min_periods=20).kurt()
            rows.append(feat)
        feats = pd.concat(rows, ignore_index=True)
        feats = feats.dropna(subset=[f'ma{max_w}', f'vol{max_w}']).reset_index(drop=True)
        return feats

    def add_cross_sectional_feats(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        byday = out.groupby('date_id', observed=True)
        # 截面 rank / z
        for col in ['z','ret1','vol20']:
            if col in out.columns:
                out[f'{col}_xrank'] = byday[col].rank(pct=True, method='average')
        for col in ['z','ret1','ma10']:
            if col in out.columns:
                out[f'{col}_xz'] = byday[col].transform(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-6))
        # 市场均值 + 与市场滚动相关
        out['mkt_z'] = byday['z'].transform('mean')
        out[f'corr_mkt_{self.corr_w}'] = out.groupby('target', observed=True, group_keys=False)\
            .apply(lambda g: g['z'].rolling(self.corr_w, min_periods=self.corr_w).corr(g['mkt_z']))
        # 自相关
        out['ret1_l1'] = out.groupby('target', observed=True)['ret1'].shift(1)
        out['ac1_20']  = out.groupby('target', observed=True, group_keys=False)\
            .apply(lambda g: g['ret1'].rolling(20, min_periods=20).corr(g['ret1_l1']))
        return out

    def _fit_pca_kmeans(self, feats_train: pd.DataFrame):
        # 用 z_xz 的 date×target 矩阵做静态 loading（训练期）
        mat = feats_train.pivot(index='date_id', columns='target', values='z_xz').fillna(0.0)
        targets = mat.columns.tolist()
        X = mat.values.astype('float32')
        if X.shape[1] < 3:
            self.pca_components_ = None; self.kmeans_centers_ = None
            self.target_order_ = None; self.target_cluster_ = None
            return
        pca = PCA(n_components=min(self.pca_k, X.shape[1]-1), random_state=self.seed)
        pca.fit(X)
        comps = pca.components_.astype('float32')      # (K, N_targets)
        loads = comps.T                                 # (N_targets, K)
        k = min(self.kmeans_k, max(2, X.shape[1]//5))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        clusters = km.fit_predict(loads)
        self.pca_components_ = comps
        self.kmeans_centers_ = km.cluster_centers_.astype('float32')
        self.target_order_   = targets
        self.target_cluster_ = {t:int(c) for t,c in zip(targets, clusters)}

    def _attach_static_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        if self.pca_components_ is None or self.target_order_ is None:
            return feats
        loads = self.pca_components_.T
        df_s = pd.DataFrame(loads, index=self.target_order_, columns=[f'pca_ld_{i+1}' for i in range(loads.shape[1])])
        df_s.index.name = 'target'; df_s = df_s.reset_index()
        if self.target_cluster_ is not None:
            df_s['cluster_id'] = df_s['target'].map(self.target_cluster_)
            for k in sorted(set(self.target_cluster_.values())):
                df_s[f'clu_{k}'] = (df_s['cluster_id']==k).astype('int8')
        return feats.merge(df_s, on='target', how='left')

    def fit(self, train_df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        feats_time = self.build_features_from_pairs(train_df, pairs_df)
        feats_full = self.add_cross_sectional_feats(feats_time)
        self._fit_pca_kmeans(feats_full)
        feats_full = self._attach_static_features(feats_full)
        return feats_full

    def transform(self, df_like_train: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:
        feats_time = self.build_features_from_pairs(df_like_train, pairs_df)
        feats_full = self.add_cross_sectional_feats(feats_time)
        feats_full = self._attach_static_features(feats_full)
        return feats_full


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
    long_lab['date_id'] = long_lab['date_id'] - long_lab['lag']
    aligned = df.merge(long_lab[['date_id','target','y']], on=['date_id','target'], how='inner')
    aligned['y'] = aligned['y'].replace(CONFIG["LABELS_FILLER"], np.nan)
    aligned = aligned.dropna(subset=['y']).reset_index(drop=True)
    return aligned


# ============================================================
# F) 时间CV + 稳健特征筛选 + 评估
# ============================================================
def make_time_folds(dates: pd.Series, n_folds: int = 3):
    uniq = np.array(sorted(pd.unique(dates)))
    n = len(uniq)
    fs = max(1, n//n_folds)
    folds=[]
    for k in range(n_folds):
        va = set(uniq[k*fs : (k+1)*fs if k<n_folds-1 else n])
        tr = set(uniq[:k*fs])
        if len(tr)>0 and len(va)>0:
            folds.append((tr,va))
    return folds

def daily_spearman_sharpe(y, p, d) -> float:
    df = pd.DataFrame({'y':y,'p':p,'d':d}).replace([np.inf,-np.inf],np.nan).dropna()
    if df.empty: return 0.0
    g = df.groupby('d').apply(lambda g_: g_['y'].rank().corr(g_['p'].rank(), method='spearman')) \
         .replace([np.inf,-np.inf],np.nan).dropna()
    return float(g.mean()/(g.std(ddof=0)+1e-12)) if len(g) else 0.0

def get_feature_cols(df: pd.DataFrame) -> list:
    blacklist={'date_id','target','y','lag','mkt_z','cluster_id'}
    cols=[]
    for c in df.columns:
        if c in blacklist: continue
        if c.startswith(('ma','vol','zscore_','pca_ld_','clu_','corr_mkt_')): cols.append(c); continue
        if c in {'z','ret1','absret','ema_vol10','skew20','kurt20','ret1_l1','ac1_20'}: cols.append(c); continue
        if c.endswith(('_xrank','_xz')): cols.append(c); continue
    return sorted(cols)

def basic_feature_filters(df: pd.DataFrame, feature_cols: list, miss_thresh=0.25, corr_thresh=0.98):
    X = df[feature_cols]
    keep = [c for c in feature_cols if X[c].isna().mean() <= miss_thresh]
    X = X[keep]
    keep = [c for c in X.columns if X[c].std(ddof=0) > 1e-12]
    X = X[keep]
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
    """
    连续 y 按“日”转成 {0,1} 相关性标签：
    - 每天取 top_q 分位为 1，其余 0；
    - 强制每个日至少出现一个 1 和一个 0（避免无梯度）。
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d)
    rel = np.zeros_like(y, dtype=np.int32)
    for day in np.unique(d):
        m = (d == day)
        y_day = y[m]
        if len(y_day) < 3 or np.nanstd(y_day) == 0.0:
            rel[m] = 0
            continue
        thr = np.nanquantile(y_day, 1.0 - top_q)
        r = (y_day >= thr).astype(np.int32)
        # 强制至少两级
        if r.sum() == 0: r[np.argmax(y_day)] = 1
        if r.sum() == len(y_day): r[np.argmin(y_day)] = 0
        rel[m] = r
    return rel




def select_features_by_cv(df: pd.DataFrame, feature_cols: list, n_keep=64, n_folds=3, seed=42):
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=['y'])
    d = df['date_id'].values
    y = df['y'].values
    y_dir = (y > 0).astype(int)
    X = df[feature_cols].values
    folds = make_time_folds(df['date_id'], n_folds=n_folds)
    l1_freq = pd.Series(0.0, index=feature_cols, dtype='float64')
    gbm_gain = pd.Series(0.0, index=feature_cols, dtype='float64')

    for tr_dates, va_dates in folds:
        tr_m = df['date_id'].isin(tr_dates).values
        va_m = df['date_id'].isin(va_dates).values
        if tr_m.sum()<500 or va_m.sum()<100: 
            continue
        X_tr, y_tr, d_tr = X[tr_m], y_dir[tr_m], d[tr_m]
        # 方向：L1-LogReg
        lr = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0.0),
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000, random_state=seed)
        )
        lr.fit(X_tr, y_tr)
        coef = lr.named_steps['logisticregression'].coef_.ravel()
        mask = (np.abs(coef) > 1e-8)
        l1_freq.loc[np.array(feature_cols)[mask]] += 1.0

        # 幅度：LGBM Ranker
        X_tr2 = X[tr_m]; y_tr2 = df['y'].values[tr_m]; d_tr2 = d_tr
        # 排序后
        ord_tr = np.argsort(d_tr2, kind='mergesort')
        X_tr_s, y_tr_s, d_tr_s = X_tr2[ord_tr], y_tr2[ord_tr], d_tr2[ord_tr]
        _, group_counts = np.unique(d_tr_s, return_counts=True)

        model_rank, mode = _train_ranker_with_fallback(X_tr_s, y_tr_s, d_tr_s, group_counts, seed)
        if hasattr(model_rank, "booster_"):
           gain = pd.Series(model_rank.booster_.feature_importance(importance_type='gain'),
                     index=feature_cols, dtype='float64')
        else:
        # 回归时用 split importance
           gain = pd.Series(getattr(model_rank, "feature_importances_", np.zeros(len(feature_cols))),
                     index=feature_cols, dtype='float64')
        gbm_gain = gbm_gain.add(gain, fill_value=0.0)


    l1_norm  = (l1_freq / max(1, len(folds))).astype('float64')
    gsum     = gbm_gain.sum()
    gbm_norm = gbm_gain / (gsum if gsum>0 else 1.0)
    fused = 0.4 * l1_norm + 0.6 * gbm_norm
    rank = fused.sort_values(ascending=False)
    selected = rank.index[:n_keep].tolist()
    score_df = pd.DataFrame({
        'feature': rank.index,
        'score': rank.values,
        'l1_freq': l1_norm.loc[rank.index].values,
        'gbm_gain_norm': gbm_norm.loc[rank.index].values
    })
    return selected, score_df


# ============================================================
# G) 训练单个 lag（LR 方向 + LGBM Rank 幅度 + 等权日内融合）
# ============================================================
def winsorize_by_day(df_: pd.DataFrame, col='y', p=0.005):
    def _w(g):
        lo, hi = g[col].quantile([p, 1-p])
        g[col] = g[col].clip(lo, hi)
        return g
    return df_.groupby('date_id', observed=True, group_keys=False).apply(_w)

def _filter_invalid_groups_for_rank(X, y_cont, y_rel, d):
    """去掉只有 1 条样本 or 只有单一等级 的日期组。"""
    df = pd.DataFrame({'d': d, 'rel': y_rel})
    cnt = df.groupby('d').size()
    nuniq = df.groupby('d')['rel'].nunique()
    valid_days = cnt[(cnt >= 3) & (nuniq >= 2)].index
    m = np.isin(d, valid_days)
    return X[m], y_cont[m], y_rel[m], d[m]

def _train_ranker_with_fallback(X_tr_s, y_tr_s, d_tr_s, group_counts, seed):
    # 诊断
    uniq_levels = pd.Series(make_daywise_relevance(y_tr_s, d_tr_s, top_q=0.25)).groupby(d_tr_s).nunique()
    print(f"[RANK][diag] days={len(uniq_levels)}  <2 levels: {(uniq_levels<2).mean():.2%}  "
        f"median_group={np.median(group_counts):.0f}")

    # 1) lambdarank + 二元标签
    y_rel = make_daywise_relevance(y_tr_s, d_tr_s, top_q=0.25)
    X1, y1, yrel1, d1 = _filter_invalid_groups_for_rank(X_tr_s, y_tr_s, y_rel, d_tr_s)
    _, gc1 = np.unique(d1, return_counts=True)
    if len(d1) > 1000 and gc1.min() >= 3:
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
                random_state=seed, n_jobs=-1
            )
            ranker.fit(X1, yrel1, group=gc1)
            imp = ranker.booster_.feature_importance(importance_type='gain')
            if np.sum(imp) > 0 and ranker.booster_.num_trees() > 1:
                return ranker, "lambdarank"
        except Exception as e:
            pass

    # 2) 兜底：rank_xendcg，连续非负标签
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
            random_state=seed, n_jobs=-1
        )
        ranker.fit(X_tr_s, y_nn, group=group_counts)
        imp = ranker.booster_.feature_importance(importance_type='gain')
        if np.sum(imp) > 0 and ranker.booster_.num_trees() > 1:
            return ranker, "rank_xendcg"
    except Exception as e:
        pass

    # 3) 最后兜底：回归“日内秩”
    y_rank = pd.Series(y_tr_s).groupby(d_tr_s).rank(method='average').values
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
        random_state=seed, n_jobs=-1
    )
    reg.fit(X_tr_s, y_rank)
    return reg, "reg_rank"


def train_one_lag(aligned_df: pd.DataFrame, feature_cols: list, n_folds: int = 3, seed: int = 42):
    df = aligned_df.copy()
    df = winsorize_by_day(df, col='y', p=0.005)

    y = df['y'].values
    y_dir = (y > 0).astype(int)
    d = df['date_id'].values
    X = df[feature_cols].values

    uniq = np.array(sorted(pd.unique(df['date_id'])))
    n = len(uniq); fs = max(1, n//n_folds); folds=[]
    for k in range(n_folds):
        va = set(uniq[k*fs : (k+1)*fs if k<n_folds-1 else n])
        tr = set(uniq[:k*fs])
        if len(tr)>0 and len(va)>0: folds.append((tr,va))

    oof_p = np.full(len(df), np.nan); oof_m = np.full(len(df), np.nan)
    models_lr, models_rank = [], []

    for (tr_dates, va_dates) in folds:
        tr_mask = df['date_id'].isin(tr_dates).values
        va_mask = df['date_id'].isin(va_dates).values
        if tr_mask.sum()<300 or va_mask.sum()<50: continue

        X_tr, y_tr, d_tr = X[tr_mask], y_dir[tr_mask], d[tr_mask]
        X_va             = X[va_mask]

        # 1) 方向
        lr = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0.0),
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                               max_iter=800, class_weight='balanced', random_state=seed)
        )
        lr.fit(X_tr, (df['y'].values[tr_mask] > 0).astype(int))
        oof_p[va_mask] = lr.predict_proba(X_va)[:,1]
        models_lr.append(lr)

        # 2) 幅度
        y_tr2 = df['y'].values[tr_mask]; d_tr2 = d_tr
        # 排序 & group
        ord_tr = np.argsort(d_tr2, kind='mergesort')
        X_tr_s, y_tr_s, d_tr_s = X[tr_mask][ord_tr], y_tr2[ord_tr], d_tr2[ord_tr]
        _, group_counts = np.unique(d_tr_s, return_counts=True)

        model_rank, mode = _train_ranker_with_fallback(X_tr_s, y_tr_s, d_tr_s, group_counts, seed)
        oof_m[va_mask] = model_rank.predict(X_va)
        models_rank.append(model_rank)

        try:
            y_rel = make_daywise_relevance(y_tr_s, d_tr_s, n_bins=5)

        # 每天有多少个不同等级

            df_diag = pd.DataFrame({'y_rel': y_rel, 'd': d_tr_s})
            uniq_per_day = df_diag.groupby('d')['y_rel'].nunique()
            bad_days_ratio = (uniq_per_day < 2).mean()
            print(f"[RANK][diag] days with <2 unique levels: {bad_days_ratio:.2%}  "
                           f"(median uniq={uniq_per_day.median():.1f})")

            ranker.fit(X_tr_s, y_rel, group=group_counts, sample_weight=w)
            oof_m[va_mask] = ranker.predict(X_va)
            models_rank.append(ranker)
        except Exception:
            # 兜底：回归日内秩
            y_tr_rank = pd.Series(y_tr2).groupby(d_tr2).rank(method='average').values
            reg = lgb.LGBMRegressor(
                objective='huber', huber_delta=1.0,
                learning_rate=0.05, num_leaves=31,
                min_child_samples=50, colsample_bytree=0.9,
                subsample=0.8, subsample_freq=1,
                reg_alpha=1.0, reg_lambda=2.0,
                n_estimators=600, random_state=seed, n_jobs=-1
            )
            reg.fit(X[tr_mask], y_tr_rank)
            oof_m[va_mask] = reg.predict(X_va)
            models_rank.append(reg)

    valid = ~np.isnan(oof_p) & ~np.isnan(oof_m)
    if valid.sum()==0:
        return {'sharpe_p':0.0,'sharpe_m':0.0,'sharpe_fuse':0.0}, \
               {'feature_cols':feature_cols,'lr_models':models_lr,'rank_models':models_rank,'iso':None}, \
               {'p':oof_p,'m':oof_m,'y':y,'d':d}

    core = (2*oof_p[valid]-1) * oof_m[valid]
    iso = IsotonicRegression(y_min=-5, y_max=5, out_of_bounds='clip')
    iso.fit(core, y[valid])

    sharpe_p = daily_spearman_sharpe(y[valid], oof_p[valid], d[valid])
    sharpe_m = daily_spearman_sharpe(y[valid], oof_m[valid], d[valid])

    df_eval = pd.DataFrame({'y':y[valid],'p':oof_p[valid],'m':oof_m[valid],'d':d[valid]})
    df_eval['fuse'] = df_eval.groupby('d')['p'].rank() * df_eval.groupby('d')['m'].rank()
    sharpe_f = daily_spearman_sharpe(df_eval['y'], df_eval['fuse'], df_eval['d'])

    metrics = {'sharpe_p':sharpe_p,'sharpe_m':sharpe_m,'sharpe_fuse':sharpe_f}
    artifacts = {'feature_cols':feature_cols,'lr_models':models_lr,'rank_models':models_rank,'iso':iso}
    oof = {'p':oof_p,'m':oof_m,'y':y,'d':d}
    return metrics, artifacts, oof


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

    # pairs 严格解析
    pairs_df = parse_pairs_strict(train_df, pairs_raw)
    lag_counts = pairs_df['lag'].value_counts().sort_index().to_dict()
    print(f"[PAIR] usable targets: {len(pairs_df)}  lag_counts={lag_counts}")

    # 特征工程（训练拟合）
    fe = FeatureEngineer(
        windows=CONFIG["WINDOWS"],
        ewma_span=CONFIG["EWMA_SPAN"],
        use_kf=CONFIG["USE_KF"],
        kf_Q=CONFIG["KF_Q"], kf_R=CONFIG["KF_R"],
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
    for L in (1,2,3,4):
        tgts = pairs_df.loc[pairs_df['lag']==L, 'target']
        ds = aligned_all[aligned_all['target'].isin(tgts)].copy()
        if len(ds)==0:
            print(f"[lag={L}] empty, skip")
            continue
        print(f"[lag={L}] samples: {ds.shape}")

        feat_cols = get_feature_cols(ds)
        feat_cols = basic_feature_filters(ds, feat_cols, miss_thresh=0.25, corr_thresh=0.98)
        selected, score_df = select_features_by_cv(ds, feat_cols, n_keep=CONFIG["FS_KEEP"], n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"])
        print(f"[lag={L}] selected {len(selected)} feats → {selected[:6]}")

        metrics, artifacts, oof = train_one_lag(ds, selected, n_folds=CONFIG["N_FOLDS"], seed=CONFIG["SEED"])
        summary[L] = metrics
        print(f"[METRIC] L{L}: Sharpe(p)={metrics['sharpe_p']:.3f}  "
              f"Sharpe(m)={metrics['sharpe_m']:.3f}  Sharpe(fuse)={metrics['sharpe_fuse']:.3f}")

    print("\n[Done] 四个 lag 的线下指标：")
    for L in (1,2,3,4):
        if L in summary:
            m = summary[L]
            print(f"  L{L}: p={m['sharpe_p']:.3f}, m={m['sharpe_m']:.3f}, fuse={m['sharpe_fuse']:.3f}")
        else:
            print(f"  L{L}: (no data)")
    return dict(summary=summary)
