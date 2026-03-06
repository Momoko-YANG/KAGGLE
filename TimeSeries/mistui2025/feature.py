import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer:
    """
    高维时间序列特征工程器
    """

    def __init__(self, window_sizes=[5, 10, 20, 60],
                 feature_groups=None,
                 target_cols=None):
        """
        Parameters:
        -----------
        window_sizes : list
            滚动窗口大小列表
        feature_groups : dict
            特征分组字典
        target_cols : list
            目标列名列表
        """
        self.window_sizes = window_sizes
        self.feature_groups = feature_groups
        self.target_cols = target_cols
        self.feature_importance = {}
        self.pca_transformers = {}
        self.imputers = {}
        self.scalers = {}

    def create_temporal_features(self, df, columns=None):
        """
        创建时间相关特征

        Returns:
        --------
        df_temporal : DataFrame
            包含时间特征的数据框
        """
        print("创建时间特征...")
        df_result = df.copy()

        if columns is None:
            columns = [col for col in df.columns
                       if col not in ['date_id'] and not col.endswith('_is_active')]

        new_features = {}

        for window in self.window_sizes:
            print(f"  处理窗口大小: {window}")

            for col in columns:
                if col in df.columns:
                    # 1. 滚动统计量（严格避免数据泄露）
                    # 均值
                    new_features[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=window).mean()

                    # 标准差（波动性）
                    new_features[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=window).std()

                    # 最大值和最小值
                    new_features[f'{col}_max_{window}'] = df[col].rolling(window=window, min_periods=window).max()
                    new_features[f'{col}_min_{window}'] = df[col].rolling(window=window, min_periods=window).min()

                    # 滚动峰值和谷值
                    new_features[f'{col}_rolling_peak_trough_{window}'] = df[col].rolling(window=window, min_periods=window).max() - df[
                        col].rolling(window=window, min_periods=window).min()

                    # 滚动中位数
                    new_features[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window, min_periods=window).median()

                    # 滚动自相关
                    new_features[f'{col}_rolling_autocorr_{window}'] = df[col].rolling(window=window, min_periods=window).apply(
                        lambda x: x.autocorr())

                    # 2. 变化率特征
                    # 简单收益率
                    new_features[f'{col}_return_{window}'] = df[col].pct_change(periods=window)
                    # 对数收益率
                    new_features[f'{col}_log_return_{window}'] = np.log(df[col] / df[col].shift(window))
                    # 移动平均收益率
                    new_features[f'{col}_moving_avg_return_{window}'] = df[col].pct_change().rolling(
                        window=window, min_periods=window).mean()
                    # 指数加权平均收益率
                    new_features[f'{col}_ewma_{window}'] = df[col].ewm(span=window).mean().pct_change()
                    # 波动率
                    new_features[f'{col}_volatility_{window}'] = np.log(df[col] / df[col].shift(1)).rolling(
                        window=window, min_periods=window).std()

                    # 3. 技术指标
                    # 相对位置（当前值在窗口中的相对位置）
                    rolling_max = df[col].rolling(window=window, min_periods=window).max()
                    rolling_min = df[col].rolling(window=window, min_periods=window).min()
                    range_val = rolling_max - rolling_min
                    range_val[range_val == 0] = 1  # 避免除零
                    new_features[f'{col}_rsi_{window}'] = (df[col] - rolling_min) / range_val

        # 4. Lag特征（更短期的）
        print("  创建Lag特征...")
        for col in columns:
            if col in df.columns:
                for lag in [1, 2, 3, 4, 5, 6, 7]:
                    new_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    new_features[f'{col}_lag_{lag}_return'] = df[col].pct_change(periods=lag).shift(lag)  # 滞后收益率特征
                    new_features[f'{col}_lag_{lag}_ma'] = df[col].shift(lag).rolling(window=20, min_periods=20).mean()  # 滞后移动平均
                    new_features[f'{col}_lag_{lag}_std'] = df[col].shift(lag).rolling(window=20, min_periods=20).std()  # 滞后标准差
                    new_features[f'{col}_trend'] = df[col] / df[col].shift(1) - 1  # 滞后特征的趋势

        # 5. 差分特征
        print("  创建差分特征...")
        for col in columns:
            if col in df.columns:
                new_features[f'{col}_diff_1'] = df[col].diff(1)
                new_features[f'{col}_diff_7'] = df[col].diff(7)
                seasonal_period = 12  # 季节性周期为12个月
                new_features[f'{col}_seasonal_diff_{seasonal_period}'] = df[col] - df[col].shift(seasonal_period)
                new_features[f'{col}_log_diff'] = np.log(df[col]) - np.log(df[col].shift(1))  # 对数差分

        # 合并新特征 - 优化：一次性合并所有特征
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df_result = pd.concat([df_result, new_features_df], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个时间特征")
        return df_result

    def apply_kalman_filter(self, series, Q=1e-5, R=0.1, model_type='local_level'):
        """
        应用卡尔曼滤波器 - 使用Local Level模型避免过度复杂
        
        Parameters:
        -----------
        series : pd.Series
            输入时间序列
        Q : float
            过程噪声协方差
        R : float
            测量噪声协方差
        model_type : str
            模型类型，目前只支持'local_level'
        """
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            print("WARNING: filterpy is not installed")
            return series, []

        if model_type == 'local_level':
            # Local Level模型: x_t = x_{t-1} + w_t, y_t = x_t + v_t
            # 状态向量只有1维：水平值
            kf = KalmanFilter(dim_x=1, dim_z=1)
            
            # 状态转移矩阵 (1x1)
            kf.F = np.array([[1.0]])
            
            # 测量矩阵 (1x1)
            kf.H = np.array([[1.0]])
            
            # 过程噪声协方差 (1x1)
            kf.Q = np.array([[Q]])
            
            # 测量噪声协方差 (1x1)
            kf.R = np.array([[R]])
            
            # 初始状态估计
            initial_value = series.iloc[0] if not pd.isna(series.iloc[0]) else 0
            kf.x = np.array([initial_value])
            
            # 初始协方差矩阵
            kf.P = np.array([[100.0]])
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 应用滤波（只使用filter，不使用smoother避免未来泄露）
        filtered_values = []
        innovations = []  # 存储创新序列用于参数优化
        
        for value in series.ffill().bfill().fillna(0):
            # 预测步骤
            kf.predict()
            
            # 更新步骤
            kf.update(value)
            
            # 记录滤波后的状态值
            filtered_values.append(kf.x[0])
            
            # 记录创新序列（用于参数优化）
            innovation = value - kf.H @ kf.x
            innovations.append(innovation[0])

        return pd.Series(filtered_values, index=series.index), innovations

    def optimize_kalman_filter(self, series, target_name=None, 
                           Q_range=[1e-6, 1e-5, 1e-4, 1e-3],
                           R_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0],
                           cv_folds=3):
        """
        优化卡尔曼滤波器参数 - 使用创新序列负对数似然作为目标函数
        
        Parameters:
        -----------
        series : pd.Series
            原始价差序列
        target_name : str
            目标名称，用于参数持久化
        Q_range : list
            过程噪声协方差的搜索范围
        R_range : list
            测量噪声协方差的搜索范围
        cv_folds : int
            交叉验证折数
        
        Returns:
        --------
        best_Q : float
            最优的Q值
        best_R : float
            最优的R值
        best_score : float
            最优得分（负对数似然）
        """
        import itertools
        import numpy as np
        
        print(f"开始优化卡尔曼滤波器参数（目标: {target_name or 'unknown'}）...")
        best_score = float('inf')
        best_Q = Q_range[0]
        best_R = R_range[0]
        
        # 确保有足够的数据
        if len(series) < cv_folds * 20:
            print("  警告：数据点太少，使用默认参数")
            return Q_range[0], R_range[0], float('inf')
        
        # 时间序列交叉验证
        n = len(series)
        fold_size = n // cv_folds
        
        for Q, R in itertools.product(Q_range, R_range):
            try:
                cv_scores = []
                
                # 交叉验证
                for fold in range(cv_folds):
                    # 训练窗口：从开始到当前折的结束
                    train_end = (fold + 1) * fold_size
                    if train_end > n - fold_size:  # 确保验证集有足够数据
                        break
                        
                    train_idx = slice(0, train_end)
                    val_idx = slice(train_end, min(train_end + fold_size, n))
                    
                    train_data = series.iloc[train_idx].ffill().bfill().fillna(0)
                    val_data = series.iloc[val_idx].ffill().bfill().fillna(0)
                    
                    if len(train_data) < 20 or len(val_data) < 5:
                        continue
                    
                    # 在训练集上拟合卡尔曼滤波器
                    kf_train, train_innovations = self.apply_kalman_filter(train_data, Q=Q, R=R)
                    
                    # 在验证集上应用相同参数的卡尔曼滤波器
                    kf_val, val_innovations = self.apply_kalman_filter(val_data, Q=Q, R=R)
                    
                    # 计算创新序列负对数似然（仅使用训练集）
                    if len(train_innovations) > 10:
                        # 计算创新序列的方差
                        innovation_var = np.var(train_innovations)
                        if innovation_var > 1e-12:
                            # 负对数似然
                            nll = 0.5 * len(train_innovations) * (np.log(2 * np.pi * innovation_var) + 1)
                        else:
                            nll = 1000  # 惩罚值
                    else:
                        nll = 1000  # 惩罚值
                    
                    cv_scores.append(nll)
                
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_Q = Q
                        best_R = R
                        
            except Exception as e:
                # 如果某个参数组合失败，继续尝试其他组合
                print(f"    参数 Q={Q}, R={R} 失败: {str(e)[:50]}")
                continue
        
        print(f"  最优参数: Q={best_Q:.2e}, R={best_R:.2f}, NLL={best_score:.4f}")
        
        # 持久化参数（如果提供了target_name）
        if target_name:
            self._save_kalman_params(target_name, best_Q, best_R, best_score)
        
        return best_Q, best_R, best_score

    def _save_kalman_params(self, target_name, Q, R, score):
        """
        保存卡尔曼滤波器参数到文件
        
        Parameters:
        -----------
        target_name : str
            目标名称
        Q : float
            过程噪声协方差
        R : float
            测量噪声协方差
        score : float
            优化得分
        """
        import json
        import os
        
        # 创建参数字典
        params = {
            'target_name': target_name,
            'Q': Q,
            'R': R,
            'score': score,
            'model_type': 'local_level'
        }
        
        # 保存到文件
        params_file = 'kalman_params.json'
        
        # 如果文件存在，读取现有参数
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    all_params = json.load(f)
            except:
                all_params = {}
        else:
            all_params = {}
        
        # 更新参数
        all_params[target_name] = params
        
        # 保存回文件
        with open(params_file, 'w') as f:
            json.dump(all_params, f, indent=2)
        
        print(f"    参数已保存: {target_name} -> Q={Q:.2e}, R={R:.2f}")

    def _load_kalman_params(self, target_name):
        """
        从文件加载卡尔曼滤波器参数
        
        Parameters:
        -----------
        target_name : str
            目标名称
        
        Returns:
        --------
        Q : float or None
            过程噪声协方差
        R : float or None
            测量噪声协方差
        """
        import json
        import os
        
        params_file = 'kalman_params.json'
        
        if not os.path.exists(params_file):
            return None, None
        
        try:
            with open(params_file, 'r') as f:
                all_params = json.load(f)
            
            if target_name in all_params:
                params = all_params[target_name]
                return params['Q'], params['R']
            else:
                return None, None
                
        except:
            return None, None

    def optimize_kalman_filter_cv(self, series, cv_folds=5,
                             Q_range=[1e-6, 1e-5, 1e-4, 1e-3],
                             R_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0]):
        """
        使用交叉验证优化卡尔曼滤波器参数（更严格的方法）
    
    :param series: 原始价差序列
    :param cv_folds: 交叉验证折数
    :param Q_range: 过程噪声协方差的搜索范围
        :param R_range: 测量噪声协方差的搜索范围
        """
        import itertools
        
        print(f"使用 {cv_folds} 折交叉验证优化卡尔曼滤波器...")
        
        # 确保有足够的数据
        if len(series) < cv_folds * 20:
            print("  警告：数据点太少，使用简单验证")
            return self.optimize_kalman_filter(series, None, None, Q_range, R_range)
        
        best_score = float('inf')
        best_Q = Q_range[0]
        best_R = R_range[0]
        
        # 时间序列交叉验证 - 使用滑窗验证避免未来泄露
        n = len(series)
        fold_size = n // cv_folds
        
        for Q, R in itertools.product(Q_range, R_range):
            cv_scores = []
            
            try:
                # 滑窗交叉验证
                for fold in range(cv_folds):
                    # 训练窗口：从开始到当前折的结束
                    train_end = (fold + 1) * fold_size
                    if train_end > n - fold_size:  # 确保验证集有足够数据
                        break
                        
                    train_idx = slice(0, train_end)
                    val_idx = slice(train_end, min(train_end + fold_size, n))
                    
                    train_data = series.iloc[train_idx].ffill().bfill().fillna(0)
                    val_data = series.iloc[val_idx].ffill().bfill().fillna(0)
                    
                    if len(train_data) < 20 or len(val_data) < 5:
                        continue
                    
                    # 在训练集上拟合
                    kf_train = self.apply_kalman_filter(train_data, Q=Q, R=R)
                    kf_val = self.apply_kalman_filter(val_data, Q=Q, R=R)
                    
                    # 计算验证分数（仅使用历史信息）
                    val_error = np.mean((val_data - kf_val) ** 2)
                    train_smoothness = np.var(kf_train.diff().diff().dropna())
                    
                    fold_score = 0.7 * val_error + 0.3 * train_smoothness
                    cv_scores.append(fold_score)
                
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    if avg_score < best_score:
                        best_score = avg_score
                        best_Q = Q
                        best_R = R
                        
            except Exception as e:
                continue
        
        print(f"  CV最优参数: Q={best_Q:.2e}, R={best_R:.2f}, CV Score={best_score:.4f}")
        return best_Q, best_R, best_score

    def create_cross_sectional_features(self, df, feature_groups=None, pairs_file='target_pairs.csv',
                                        use_kalman=False, optimize_kalman=False,
                                        Q_range=[1e-6, 1e-5, 1e-4, 1e-3],
                                        R_range=[0.01, 0.05, 0.1, 0.15, 0.5, 1.0]):
        """
        创建横截面特征（不同资产间的关系）

        Parameters:
        -----------
        df : DataFrame
            输入数据®
        feature_groups : dict
            特征分组
        pairs_file : str
            输入数据
        use_kalman : bool
            是否使用卡尔曼滤波
        optimize_kalman : bool
            是否优化卡尔曼滤波
        Q_range : list
            卡尔曼滤波Q范围
        R_range : list
            卡尔曼滤波R范围
        """
        print("\n创建横截面特征...")
        df_result = df.copy()

        if feature_groups is None:
            feature_groups = self._auto_group_features(df)

        new_features = {}

        # 1. 创建组内统计特征（同尺度化后）
        print("  创建组内统计特征...")
        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols if col in df.columns]

            if len(valid_cols) > 1:
                # 先进行同尺度化处理
                df_normalized = self._normalize_cross_sectional_features(df[valid_cols], method='zscore')
                
                # 计算组内平均值（同尺度化后）
                new_features[f'{group_name}_mean'] = df_normalized.mean(axis=1)

                # 计算组内标准差（同尺度化后）
                new_features[f'{group_name}_std'] = df_normalized.std(axis=1)

                # 计算组内最大值和最小值（同尺度化后）
                new_features[f'{group_name}_max'] = df_normalized.max(axis=1)
                new_features[f'{group_name}_min'] = df_normalized.min(axis=1)

                # 计算组内排名（基于原始值，但使用同尺度化后的数据）
                for col in valid_cols[:5]:  # 计算组内排名
                    if col in df_normalized.columns:
                        rank = df_normalized.rank(axis=1, pct=True)[col]
                        new_features[f'{col}_rank_in_{group_name}'] = rank

        # 2. 创建跨组关系特征（同尺度化后）
        print("  创建跨组关系特征...")
        group_means = {}
        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols if col in df.columns]
            if valid_cols:
                # 先进行同尺度化，再计算平均值
                df_normalized = self._normalize_cross_sectional_features(df[valid_cols], method='zscore')
                group_means[group_name] = df_normalized.mean(axis=1)

        # 计算跨组比值（基于同尺度化后的数据）
        if 'JPX' in group_means and 'US_Stock' in group_means:
            new_features['JPX_US_ratio'] = group_means['JPX'] / (group_means['US_Stock'] + 1e-8)

        if 'FX' in group_means and 'US_Stock' in group_means:
            new_features['FX_US_ratio'] = group_means['FX'] / (group_means['US_Stock'] + 1e-8)

        # 3. 创建滚动相关性与卡尔曼滤波价差特征
        print("  创建滚动相关性与卡尔曼滤波价差特征...")

        # 读取输入数据
        key_pairs = []
        spread_pairs = []
        try:
            import os
            if os.path.exists(pairs_file):
                pairs_df = pd.read_csv(pairs_file)
                print(f"    从 {pairs_file} 读取了 {len(pairs_df)} 个资产对")

                # 读取输入数据
                # 1. 一个资产: "US_Stock_VT_adj_close"
                # 2. 两个资产的差: "LME_PB_Close - US_Stock_VT_adj_close"
                for idx, row in pairs_df.iterrows():
                    pair_str = row['pair'].strip()

                    # 检查是否包含减号（表示两个资产的差）
                    if ' - ' in pair_str:
                        # 分割成两个资产
                        asset1, asset2 = pair_str.split(' - ')
                        asset1 = asset1.strip()
                        asset2 = asset2.strip()

                        # 检查这两个资产是否在数据列中存在
                        if asset1 in df.columns and asset2 in df.columns:
                            key_pairs.append((asset1, asset2))
                        else:
                            # 尝试模糊匹配
                            cols1 = [col for col in df.columns if asset1 in col or col in asset1]
                            cols2 = [col for col in df.columns if asset2 in col or col in asset2]
                            if cols1 and cols2:
                                spread_pairs.append((cols1[0], cols2[0], idx))
                                key_pairs.append((cols1[0], cols2[0]))


                    else:
                        # 单个资产的情况，需要找另一个相关资产配对
                        # 这里我们可以选择与基准资产配对，比如VT（全球股票指数）
                        if pair_str in df.columns:
                            # 选择一个基准资产
                            benchmark_cols = ['US_Stock_VT_adj_close', 'FX_USDJPY', 'US_Stock_GLD_adj_close']
                            for benchmark in benchmark_cols:
                                if benchmark in df.columns and benchmark != pair_str:
                                    key_pairs.append((pair_str, benchmark))
                                    break

                # 去重
                key_pairs = list(set(key_pairs))
                print(f"    成功构建了 {len(key_pairs)} 个唯一资产对")
                print(f"    其中 {len(spread_pairs)} 个价差对")

            else:
                print(f"    警告：未找到 {pairs_file}，使用默认资产对")
                # 使用默认资产对
                key_pairs = [
                    ('US_Stock_GLD_adj_close', 'FX_USDJPY'),
                    ('US_Stock_XLE_adj_close', 'US_Stock_CVX_adj_close'),
                    ('LME_AH_Close', 'US_Stock_FCX_adj_close'),
                ]

        except Exception as e:
            print(f"    读取资产对文件时出错: {e}")
            print("    使用默认资产对")
            key_pairs = [
                ('US_Stock_GLD_adj_close', 'FX_USDJPY'),
                ('US_Stock_XLE_adj_close', 'US_Stock_CVX_adj_close'),
                ('LME_AH_Close', 'US_Stock_FCX_adj_close'),
            ]

        # 创建价差特征
        if use_kalman:
            print("应用卡尔曼滤波到价差")

            try:
                from filterpy.kalman import KalmanFilter
                kalman_available = True
            except ImportError:
                print("filterpy未安装，跳过卡尔曼滤波，请运行 pip install filterpy")
                kalman_available = False
        else:
            kalman_available = False

        kalman_params = {}

        for i, (asset1, asset2, idx) in enumerate(spread_pairs[:50]): # 限制数量避免过长
            if asset1 in df.columns and asset2 in df.columns:
                # 先对两个资产进行同尺度化，再计算价差
                asset_pair_data = df[[asset1, asset2]].copy()
                asset_pair_normalized = self._normalize_cross_sectional_features(asset_pair_data, method='zscore')
                spread = asset_pair_normalized[asset1] - asset_pair_normalized[asset2]

                if kalman_available and use_kalman:
                    # 创建目标名称用于参数持久化
                    target_name = f'spread_{asset1}_{asset2}'
                    
                    # 尝试加载已保存的参数
                    saved_Q, saved_R = self._load_kalman_params(target_name)
                    
                    if optimize_kalman and (saved_Q is None or saved_R is None):
                        if i % 10 == 0:
                            print(f"  优化第{i + 1}/{min(50, len(spread_pairs))}个价差对的卡尔曼参数")

                        best_Q, best_R, best_score = self.optimize_kalman_filter(
                            spread, target_name=target_name,
                            Q_range=Q_range, R_range=R_range
                        )
                        kalman_params[f'{asset1}_{asset2}'] = (best_Q, best_R)
                    elif saved_Q is not None and saved_R is not None:
                        # 使用已保存的参数
                        kalman_params[f'{asset1}_{asset2}'] = (saved_Q, saved_R)
                        print(f"    使用已保存参数: {target_name} -> Q={saved_Q:.2e}, R={saved_R:.2f}")
                    else:
                        # 使用默认参数
                        kalman_params[f'{asset1}_{asset2}'] = (Q_range[0], R_range[0])

        # 创建滚动相关性特征
        correlation_count = 0
        max_correlations = 100  # 限制最大相关性特征数量，避免特征爆炸

        for i, (col1, col2) in enumerate(key_pairs):
            if correlation_count >= max_correlations:
                print(f"    达到最大相关性特征数量限制 ({max_correlations})，停止创建")
                break

            if col1 in df.columns and col2 in df.columns:
                # 先对两个资产进行同尺度化，再计算相关性
                asset_pair_data = df[[col1, col2]].copy()
                asset_pair_normalized = self._normalize_cross_sectional_features(asset_pair_data, method='rolling_zscore', window=60)
                
                for window in [20, 60]:
                    corr = asset_pair_normalized[col1].rolling(window=window, min_periods=window).corr(asset_pair_normalized[col2])
                    # 创建相关性特征
                    feature_name = f'pair_{i}_corr_{window}'
                    new_features[feature_name] = corr
                    correlation_count += 1

        print(f"    创建了 {correlation_count} 个相关性特征")

        # 合并新特征 - 优化：一次性合并所有特征
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df_result = pd.concat([df_result, new_features_df], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个横截面特征")
        return df_result

    def create_market_regime_features(self, df):
        """
        创建市场状态特征
        """
        print("\n创建市场状态特征...")
        df_result = df.copy()
        new_features = {}

        # 1. 波动率regime（同尺度化后）
        volatility_cols = [col for col in df.columns if 'std_' in col]
        if volatility_cols:
            # 先进行同尺度化，再计算市场整体波动率
            vol_normalized = self._normalize_cross_sectional_features(df[volatility_cols], method='zscore')
            new_features['market_volatility'] = vol_normalized.mean(axis=1)

            # 波动率regime（高/中/低）
            vol_percentiles = vol_normalized.mean(axis=1).quantile([0.33, 0.67])
            new_features['volatility_regime'] = pd.cut(
                vol_normalized.mean(axis=1),
                bins=[-np.inf, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
                labels=[0, 1, 2]
            ).astype(float)

        # 2. 趋势强度
        # 使用移动平均的斜率
        ma_cols = [col for col in df.columns if '_ma_20' in col]
        if ma_cols:
            for col in ma_cols[:10]:  # 限制数量
                # 20日均线的5日变化率
                ma_change = df[col].diff(5) / (df[col].shift(5) + 1e-8)
                new_features[f'{col}_trend_strength'] = ma_change

        # 3. 市场情绪指标（同尺度化后）
        # 使用避险资产vs风险资产的比率
        safe_assets = ['US_Stock_GLD_adj_close', 'US_Stock_IAU_adj_close', 'FX_USDJPY']
        risk_assets = ['US_Stock_XLE_adj_close', 'US_Stock_FCX_adj_close']

        safe_cols = [col for col in safe_assets if col in df.columns]
        risk_cols = [col for col in risk_assets if col in df.columns]

        if safe_cols and risk_cols:
            # 先进行同尺度化，再计算比值
            safe_normalized = self._normalize_cross_sectional_features(df[safe_cols], method='zscore')
            risk_normalized = self._normalize_cross_sectional_features(df[risk_cols], method='zscore')
            
            safe_mean = safe_normalized.mean(axis=1)
            risk_mean = risk_normalized.mean(axis=1)
            new_features['risk_on_off_ratio'] = risk_mean / (safe_mean + 1e-8)

        # 合并新特征
        df_result = pd.concat([df_result, pd.DataFrame(new_features)], axis=1)

        print(f"  ✓ 创建了 {len(new_features)} 个市场状态特征")
        return df_result

    def create_pca_features(self, df, n_components=50, feature_groups=None, 
                        train_end_idx=None, fit_on_train_only=True):
        """
        使用PCA降维创建主成分特征，避免数据泄露
    
    Parameters:
    -----------
    df : DataFrame
        输入数据
    n_components : int
        保留的主成分数量
    feature_groups : dict
        特征分组
    train_end_idx : int
        训练集结束位置（用于避免数据泄露）
    fit_on_train_only : bool
        是否只在训练集上拟合变换器
        """
        print(f"\n创建PCA特征 (保留{n_components}个主成分)...")
        df_result = df.copy()

        if feature_groups is None:
            feature_groups = self._auto_group_features(df)

        # 确定训练集边界
        if train_end_idx is None:
            if fit_on_train_only:
                # 默认使用80%作为训练集
                train_end_idx = int(len(df) * 0.8)
                print(f"  使用前{train_end_idx}行作为训练集拟合变换器")
            else:
                # 如果不要求避免泄露，使用全部数据
                train_end_idx = len(df)
                print("  警告：在全部数据上拟合变换器（可能存在数据泄露）")

        pca_features = {}
        self.pca_transformers = {}  # 保存变换器用于后续的测试数据

        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols
                          if col in df.columns and not col.endswith('_is_active')]

            if len(valid_cols) > 3:  # 只对有足够特征的组做PCA
                print(f"  处理 {group_name} 组 ({len(valid_cols)} 个特征)...")

            try:
                # 1. 准备训练数据（只使用训练集）
                train_data = df[valid_cols].iloc[:train_end_idx].copy()
                
                # 处理缺失值 - 在训练集上计算填充值
                train_fill_values = train_data.median()  # 使用中位数填充
                train_data_filled = train_data.fillna(train_fill_values)
                
                # 检查是否有足够的数据点
                if len(train_data_filled) < max(10, len(valid_cols)):
                    print(f"    跳过 {group_name}：训练数据不足")
                    continue

                # 2. 在训练集上拟合标准化器
                scaler = StandardScaler()
                train_data_scaled = scaler.fit_transform(train_data_filled)

                # 3. 在训练集上拟合PCA
                n_comp = min(n_components, len(valid_cols), len(train_data_filled))
                pca = PCA(n_components=n_comp)
                train_pca_result = pca.fit_transform(train_data_scaled)

                # 4. 对全部数据进行变换（使用训练集拟合的变换器）
                full_data = df[valid_cols].copy()
                full_data_filled = full_data.fillna(train_fill_values)  # 使用训练集的填充值
                
                # 应用训练集拟合的标准化器
                full_data_scaled = scaler.transform(full_data_filled)
                
                # 应用训练集拟合的PCA
                full_pca_result = pca.transform(full_data_scaled)

                # 5. 保存主成分特征
                for i in range(min(10, n_comp)):  # 每组最多保留10个主成分
                    pca_features[f'{group_name}_PC{i + 1}'] = full_pca_result[:, i]

                # 6. 保存变换器（用于后续测试数据）
                self.pca_transformers[group_name] = {
                    'scaler': scaler,
                    'pca': pca,
                    'fill_values': train_fill_values,
                    'feature_cols': valid_cols
                }

                # 7. 输出解释方差比例
                explained_var = pca.explained_variance_ratio_[:min(10, n_comp)].sum()
                print(f"    前{min(10, n_comp)}个主成分解释了 {explained_var:.2%} 的方差")
                
                # 8. 检查训练集vs全集的PCA结果一致性
                if fit_on_train_only:
                    train_pca_mean = np.mean(train_pca_result[:, 0])
                    full_train_pca_mean = np.mean(full_pca_result[:train_end_idx, 0])
                    consistency_check = abs(train_pca_mean - full_train_pca_mean)
                    if consistency_check > 1e-10:
                        print(f"    警告：变换一致性检查失败 ({consistency_check:.2e})")

            except Exception as e:
                print(f"    处理 {group_name} 组时出错: {str(e)[:100]}")
                continue

        # 合并PCA特征
        if pca_features:
            pca_features_df = pd.DataFrame(pca_features, index=df.index)
            df_result = pd.concat([df_result, pca_features_df], axis=1)

        print(f"  ✓ 创建了 {len(pca_features)} 个PCA特征")
        return df_result

    def transform_new_data_pca(self, df_new, feature_groups=None):
        """
        对新数据应用已拟合的PCA变换器（用于测试集）
    
    Parameters:
    -----------
    df_new : DataFrame
        新的数据（如测试集）
    feature_groups : dict
        特征分组（可选，如果None则使用保存的变换器信息）
    """
        print("\n对新数据应用PCA变换...")
        
        if not hasattr(self, 'pca_transformers') or not self.pca_transformers:
            print("  错误：未找到已拟合的PCA变换器，请先运行create_pca_features")
            return df_new.copy()
        
        df_result = df_new.copy()
        pca_features = {}
        
        for group_name, transformer_info in self.pca_transformers.items():
            try:
                # 提取保存的变换器和参数
                scaler = transformer_info['scaler']
                pca = transformer_info['pca']
                fill_values = transformer_info['fill_values']
                feature_cols = transformer_info['feature_cols']
                
                # 检查新数据是否包含所需特征
                available_cols = [col for col in feature_cols if col in df_new.columns]
                if len(available_cols) != len(feature_cols):
                    print(f"    警告：{group_name} 组缺少部分特征，跳过")
                    continue
                
                # 准备新数据
                new_data = df_new[feature_cols].copy()
                new_data_filled = new_data.fillna(fill_values)  # 使用训练时的填充值
                
                # 应用标准化（使用训练时拟合的scaler）
                new_data_scaled = scaler.transform(new_data_filled)
                
                # 应用PCA（使用训练时拟合的pca）
                new_pca_result = pca.transform(new_data_scaled)
                
                # 保存变换结果
                n_components_saved = min(10, new_pca_result.shape[1])
                for i in range(n_components_saved):
                    pca_features[f'{group_name}_PC{i + 1}'] = new_pca_result[:, i]
                    
                print(f"    ✓ 成功变换 {group_name} 组")
                
            except Exception as e:
                print(f"    处理 {group_name} 组时出错: {str(e)[:100]}")
                continue
        
        # 合并PCA特征
        if pca_features:
            pca_features_df = pd.DataFrame(pca_features, index=df_new.index)
            df_result = pd.concat([df_result, pca_features_df], axis=1)
        
        print(f"  ✓ 为新数据创建了 {len(pca_features)} 个PCA特征")
        return df_result

    def create_pca_features_with_cv(self, df, n_components=50, feature_groups=None, 
                               cv_folds=5, return_oof_predictions=False):
        """
    使用交叉验证创建PCA特征，完全避免数据泄露
    
    Parameters:
    -----------
    df : DataFrame
        输入数据
    n_components : int
        保留的主成分数量
    feature_groups : dict
        特征分组
    cv_folds : int
        交叉验证折数
    return_oof_predictions : bool
        是否返回out-of-fold预测
        """
        print(f"\n使用{cv_folds}折交叉验证创建PCA特征...")
        
        if feature_groups is None:
            feature_groups = self._auto_group_features(df)
        
        n = len(df)
        fold_size = n // cv_folds
        
        # 初始化结果容器
        pca_features = {}
        oof_features = {}  # out-of-fold特征
        
        # 为每个组创建特征容器
        for group_name, group_cols in feature_groups.items():
            valid_cols = [col for col in group_cols
                          if col in df.columns and not col.endswith('_is_active')]
            
            if len(valid_cols) > 3:
                n_comp = min(n_components, len(valid_cols), fold_size)
                for i in range(min(10, n_comp)):
                    pca_features[f'{group_name}_PC{i + 1}'] = np.zeros(n)
                    oof_features[f'{group_name}_PC{i + 1}'] = np.zeros(n)
        
        # 交叉验证
        for fold in range(cv_folds):
            print(f"  处理第 {fold + 1}/{cv_folds} 折...")
            
            # 确定训练集和验证集
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n
            
            train_indices = list(range(0, val_start)) + list(range(val_end, n))
            val_indices = list(range(val_start, val_end))
            
            # 对每个组分别处理
            for group_name, group_cols in feature_groups.items():
                valid_cols = [col for col in group_cols
                              if col in df.columns and not col.endswith('_is_active')]
                
                if len(valid_cols) <= 3:
                    continue
                    
                try:
                    # 准备训练和验证数据
                    train_data = df[valid_cols].iloc[train_indices].copy()
                    val_data = df[valid_cols].iloc[val_indices].copy()
                    
                    # 处理缺失值
                    train_fill_values = train_data.median()
                    train_data_filled = train_data.fillna(train_fill_values)
                    val_data_filled = val_data.fillna(train_fill_values)
                    
                    # 拟合变换器（仅在训练集上）
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_data_filled)
                    
                    n_comp = min(n_components, len(valid_cols), len(train_data_filled))
                    pca = PCA(n_components=n_comp)
                    pca.fit(train_scaled)
                    
                    # 变换验证集
                    val_scaled = scaler.transform(val_data_filled)
                    val_pca = pca.transform(val_scaled)
                    
                    # 保存out-of-fold结果
                    for i in range(min(10, n_comp)):
                        feature_name = f'{group_name}_PC{i + 1}'
                        oof_features[feature_name][val_indices] = val_pca[:, i]
                        
                except Exception as e:
                    print(f"    第{fold + 1}折处理{group_name}组时出错: {str(e)[:50]}")
                    continue
        
        # 创建最终的特征DataFrame
        if return_oof_predictions:
            # 返回out-of-fold预测
            result_features = oof_features
        else:
            # 在全数据上重新拟合（用于最终模型）
            print("  在全数据上重新拟合最终变换器...")
            result_features = {}
            
            for group_name, group_cols in feature_groups.items():
                valid_cols = [col for col in group_cols
                              if col in df.columns and not col.endswith('_is_active')]
                
                if len(valid_cols) <= 3:
                    continue
                    
                try:
                    # 使用80%数据拟合最终变换器
                    train_end = int(len(df) * 0.8)
                    train_data = df[valid_cols].iloc[:train_end].copy()
                    
                    train_fill_values = train_data.median()
                    train_data_filled = train_data.fillna(train_fill_values)
                    
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_data_filled)
                    
                    n_comp = min(n_components, len(valid_cols), len(train_data_filled))
                    pca = PCA(n_components=n_comp)
                    pca.fit(train_scaled)
                    
                    # 变换全部数据
                    full_data = df[valid_cols].fillna(train_fill_values)
                    full_scaled = scaler.transform(full_data)
                    full_pca = pca.transform(full_scaled)
                    
                    # 保存特征
                    for i in range(min(10, n_comp)):
                        feature_name = f'{group_name}_PC{i + 1}'
                        result_features[feature_name] = full_pca[:, i]
                        
                except Exception as e:
                    print(f"    最终拟合{group_name}组时出错: {str(e)[:50]}")
                    continue
    
        # 合并结果
        if result_features:
            pca_features_df = pd.DataFrame(result_features, index=df.index)
            df_result = pd.concat([df.copy(), pca_features_df], axis=1)
        else:
            df_result = df.copy()
        
        print(f"  ✓ 创建了 {len(result_features)} 个CV-PCA特征")
        return df_result

    def create_interaction_features(self, df, important_features=None, max_interactions=50):
        """
        创建重要特征间的交互项
        """
        print(f"\n创建交互特征 (最多{max_interactions}个)...")
        df_result = df.copy()

        if important_features is None:
            # 选择一些关键特征
            important_features = [
                'US_Stock_GLD_adj_close',
                'FX_USDJPY',
                'FX_CADCHF',
                'FX_NZDCAD',
                'FX_GBPCAD',
                'FX_EURUSD',
                'volume',
                'Volatility',
                'US_Stock_XLE_adj_close',
                'LME_AH_Close',
                'LME_CA_Close',
                'LME_ZS_Close',
                'LME_PB_Close',
                'JPX_Gold_Standard_Futures_Open',
                'JPX_Platinum_Standard_Futures_Open',
                'JPX_Platinum_Standard_Futures_Close',
                'JPX_Gold_Standard_Futures_Close'
            ]

        # 过滤存在的特征
        valid_features = [f for f in important_features if f in df.columns]

        interaction_features = {}
        count = 0

        # 创建乘积交互
        for i, feat1 in enumerate(valid_features):
            for feat2 in valid_features[i + 1:]:
                if count >= max_interactions:
                    break

                # 乘积
                interaction_features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]

                # 比率
                interaction_features[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)

                count += 2

        # 合并交互特征
        df_result = pd.concat([df_result, pd.DataFrame(interaction_features)], axis=1)

        print(f"  ✓ 创建了 {len(interaction_features)} 个交互特征")
        return df_result

    def _auto_group_features(self, df):
        """
        自动将特征分组
        """
        groups = {
            'LME': [],
            'JPX': [],
            'US_Stock': [],
            'FX': []
        }

        for col in df.columns:
            if col.startswith('LME'):
                groups['LME'].append(col)
            elif col.startswith('JPX'):
                groups['JPX'].append(col)
            elif col.startswith('US_Stock'):
                groups['US_Stock'].append(col)
            elif col.startswith('FX'):
                groups['FX'].append(col)

        # 进一步细分US_Stock
        groups['US_Stock_price'] = [col for col in groups['US_Stock'] if 'close' in col or 'open' in col]
        groups['US_Stock_volume'] = [col for col in groups['US_Stock'] if 'volume' in col]

        return groups

    def _normalize_cross_sectional_features(self, df_features, method='zscore', window=60):
        """
        对跨资产特征进行同尺度化处理
        
        Parameters:
        -----------
        df_features : DataFrame
            需要标准化的特征数据框
        method : str
            标准化方法 ('zscore', 'robust_zscore', 'minmax', 'rolling_zscore')
        window : int
            滚动窗口大小（用于rolling_zscore）
        
        Returns:
        --------
        df_normalized : DataFrame
            标准化后的数据框
        """
        df_normalized = df_features.copy()
        
        if method == 'zscore':
            # 当日截面z-score标准化
            for idx in df_features.index:
                row_data = df_features.loc[idx].dropna()
                if len(row_data) > 1:
                    mean_val = row_data.mean()
                    std_val = row_data.std()
                    if std_val > 1e-8:
                        df_normalized.loc[idx] = (df_features.loc[idx] - mean_val) / std_val
                    else:
                        df_normalized.loc[idx] = 0
                else:
                    df_normalized.loc[idx] = 0
                    
        elif method == 'robust_zscore':
            # 使用中位数和MAD的稳健标准化
            for idx in df_features.index:
                row_data = df_features.loc[idx].dropna()
                if len(row_data) > 1:
                    median_val = row_data.median()
                    mad_val = np.median(np.abs(row_data - median_val))
                    if mad_val > 1e-8:
                        df_normalized.loc[idx] = (df_features.loc[idx] - median_val) / (1.4826 * mad_val)
                    else:
                        df_normalized.loc[idx] = 0
                else:
                    df_normalized.loc[idx] = 0
                    
        elif method == 'minmax':
            # Min-Max标准化到[0,1]
            for idx in df_features.index:
                row_data = df_features.loc[idx].dropna()
                if len(row_data) > 1:
                    min_val = row_data.min()
                    max_val = row_data.max()
                    if max_val - min_val > 1e-8:
                        df_normalized.loc[idx] = (df_features.loc[idx] - min_val) / (max_val - min_val)
                    else:
                        df_normalized.loc[idx] = 0.5
                else:
                    df_normalized.loc[idx] = 0.5
                    
        elif method == 'rolling_zscore':
            # 历史滚动标准化
            for col in df_features.columns:
                if col in df_features.columns:
                    # 计算滚动均值和标准差
                    rolling_mean = df_features[col].rolling(window=window, min_periods=window).mean()
                    rolling_std = df_features[col].rolling(window=window, min_periods=window).std()
                    
                    # 标准化
                    df_normalized[col] = (df_features[col] - rolling_mean) / (rolling_std + 1e-8)
                    
        else:
            raise ValueError(f"未知的标准化方法: {method}")
        
        return df_normalized

    def remove_highly_correlated_features(self, df, correlation_threshold=0.98, 
                                        priority_order=None, verbose=True):
        """
        移除高度相关的特征，避免强共线性问题
        
        Parameters:
        -----------
        df : DataFrame
            包含特征的数据框
        correlation_threshold : float
            相关性阈值，超过此值的特征对将被处理
        priority_order : list
            特征优先级顺序，优先级高的特征会被保留
        verbose : bool
            是否打印详细信息
        
        Returns:
        --------
        df_cleaned : DataFrame
            清理后的数据框
        removed_features : list
            被移除的特征列表
        """
        if verbose:
            print(f"\n开始清理强共线性特征（阈值: {correlation_threshold}）...")
        
        # 获取特征列（排除date_id等非特征列）
        feature_cols = [col for col in df.columns 
                       if col not in ['date_id', 'target'] and not col.endswith('_is_active')]
        
        if len(feature_cols) <= 1:
            if verbose:
                print("  特征数量不足，跳过共线性清理")
            return df, []
        
        # 计算相关性矩阵
        if verbose:
            print(f"  计算{len(feature_cols)}个特征的相关性矩阵...")
        
        corr_matrix = df[feature_cols].corr().abs()
        
        # 设置默认优先级顺序（简单特征优先）
        if priority_order is None:
            priority_order = self._get_feature_priority_order(feature_cols)
        
        # 找到需要移除的特征
        removed_features = set()
        
        # 遍历相关性矩阵的上三角
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                feat1, feat2 = feature_cols[i], feature_cols[j]
                
                # 跳过已经移除的特征
                if feat1 in removed_features or feat2 in removed_features:
                    continue
                
                # 检查相关性
                if corr_matrix.loc[feat1, feat2] > correlation_threshold:
                    # 根据优先级决定保留哪个特征
                    if feat1 in priority_order and feat2 in priority_order:
                        # 优先级数字越小，优先级越高
                        if priority_order.index(feat1) < priority_order.index(feat2):
                            removed_features.add(feat2)
                        else:
                            removed_features.add(feat1)
                    elif feat1 in priority_order:
                        removed_features.add(feat2)
                    elif feat2 in priority_order:
                        removed_features.add(feat1)
                    else:
                        # 都不在优先级列表中，保留更简单的特征
                        if self._is_simpler_feature(feat1, feat2):
                            removed_features.add(feat2)
                        else:
                            removed_features.add(feat1)
        
        removed_features = list(removed_features)
        
        # 创建清理后的数据框
        cols_to_keep = [col for col in df.columns if col not in removed_features]
        df_cleaned = df[cols_to_keep].copy()
        
        if verbose:
            print(f"  ✓ 移除了 {len(removed_features)} 个高度相关特征")
            print(f"  剩余特征数: {len(cols_to_keep)}")
            if removed_features:
                print(f"  被移除的特征: {removed_features[:10]}{'...' if len(removed_features) > 10 else ''}")
        
        return df_cleaned, removed_features
    
    def _get_feature_priority_order(self, feature_cols):
        """
        根据特征复杂度生成优先级顺序（数字越小优先级越高）
        """
        priority_scores = {}
        
        for feat in feature_cols:
            score = 0
            
            # 原始特征优先级最高
            if not any(x in feat for x in ['_ma_', '_std_', '_max_', '_min_', '_return_', 
                                         '_lag_', '_corr_', '_PC', '_x_', '_div_', '_rank_']):
                score += 1000
            
            # 简单统计特征
            elif any(x in feat for x in ['_ma_', '_std_', '_max_', '_min_']):
                score += 800
            
            # 收益率特征
            elif '_return_' in feat:
                score += 700
            
            # Lag特征
            elif '_lag_' in feat:
                score += 600
            
            # 相关性特征
            elif '_corr_' in feat:
                score += 500
            
            # PCA特征
            elif '_PC' in feat:
                score += 400
            
            # 交互特征
            elif '_x_' in feat or '_div_' in feat:
                score += 300
            
            # 排名特征
            elif '_rank_' in feat:
                score += 200
            
            # 其他复杂特征
            else:
                score += 100
            
            # 窗口大小影响优先级（小窗口优先）
            import re
            window_match = re.search(r'_(\d+)$', feat)
            if window_match:
                window_size = int(window_match.group(1))
                score += (100 - window_size)  # 小窗口优先级高
            
            priority_scores[feat] = score
        
        # 按分数降序排列（分数高的优先级高）
        sorted_features = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_features]
    
    def _is_simpler_feature(self, feat1, feat2):
        """
        判断哪个特征更简单（用于相关性清理时的选择）
        """
        # 计算特征复杂度
        def complexity_score(feat):
            score = 0
            if '_x_' in feat or '_div_' in feat:
                score += 3  # 交互特征最复杂
            if '_PC' in feat:
                score += 2  # PCA特征复杂
            if '_corr_' in feat:
                score += 2  # 相关性特征复杂
            if '_lag_' in feat:
                score += 1  # Lag特征中等复杂
            if '_ma_' in feat or '_std_' in feat:
                score += 1  # 统计特征中等复杂
            return score
        
        return complexity_score(feat1) <= complexity_score(feat2)
    
    def create_correlation_clusters(self, df, correlation_threshold=0.95, 
                                  method='hierarchical', verbose=True):
        """
        将高度相关的特征聚类，每个簇选择一个代表特征
        
        Parameters:
        -----------
        df : DataFrame
            包含特征的数据框
        correlation_threshold : float
            聚类阈值
        method : str
            聚类方法 ('hierarchical', 'connected_components')
        verbose : bool
            是否打印详细信息
        
        Returns:
        --------
        df_clustered : DataFrame
            聚类后的数据框
        cluster_info : dict
            聚类信息
        """
        if verbose:
            print(f"\n开始特征聚类（阈值: {correlation_threshold}，方法: {method}）...")
        
        # 获取特征列
        feature_cols = [col for col in df.columns 
                       if col not in ['date_id', 'target'] and not col.endswith('_is_active')]
        
        if len(feature_cols) <= 1:
            if verbose:
                print("  特征数量不足，跳过聚类")
            return df, {}
        
        # 计算相关性矩阵
        corr_matrix = df[feature_cols].corr().abs()
        
        if method == 'hierarchical':
            clusters = self._hierarchical_clustering(corr_matrix, correlation_threshold)
        elif method == 'connected_components':
            clusters = self._connected_components_clustering(corr_matrix, correlation_threshold)
        else:
            raise ValueError(f"未知的聚类方法: {method}")
        
        # 为每个簇选择代表特征
        selected_features = []
        cluster_info = {}
        
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                selected_features.extend(cluster)
                cluster_info[f'cluster_{i}'] = {
                    'features': cluster,
                    'representative': cluster[0],
                    'size': 1
                }
            else:
                # 选择代表特征（优先级最高的）
                priority_order = self._get_feature_priority_order(cluster)
                representative = priority_order[0]
                selected_features.append(representative)
                
                cluster_info[f'cluster_{i}'] = {
                    'features': cluster,
                    'representative': representative,
                    'size': len(cluster)
                }
        
        # 创建聚类后的数据框
        cols_to_keep = [col for col in df.columns if col in selected_features or col not in feature_cols]
        df_clustered = df[cols_to_keep].copy()
        
        if verbose:
            print(f"  ✓ 将 {len(feature_cols)} 个特征聚类为 {len(clusters)} 个簇")
            print(f"  选择了 {len(selected_features)} 个代表特征")
            print(f"  压缩比例: {len(selected_features)/len(feature_cols):.2%}")
        
        return df_clustered, cluster_info
    
    def _hierarchical_clustering(self, corr_matrix, threshold):
        """
        使用层次聚类进行特征聚类
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # 将相关性转换为距离
        distance_matrix = 1 - corr_matrix.values
        
        # 层次聚类
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
        
        # 组织聚类结果
        cluster_dict = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(corr_matrix.index[i])
        
        return list(cluster_dict.values())
    
    def _connected_components_clustering(self, corr_matrix, threshold):
        """
        使用连通分量进行特征聚类
        """
        import networkx as nx
        
        # 创建图
        G = nx.Graph()
        G.add_nodes_from(corr_matrix.index)
        
        # 添加边（相关性超过阈值的特征对）
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > threshold:
                    G.add_edge(corr_matrix.index[i], corr_matrix.index[j])
        
        # 找到连通分量
        clusters = list(nx.connected_components(G))
        
        return [list(cluster) for cluster in clusters]
    
import os
from typing import Dict, List, Tuple, Union, Optional

def group_targets_by_lag_from_pairs(
    pairs: Union[str, os.PathLike, pd.DataFrame],
    labels_df: Optional[pd.DataFrame] = None,
    target_col: str = "target",
    lag_col: str = "lag",
    date_col: str = "date_id",
    allowed_lags: Tuple[int, ...] = (1, 2, 3, 4),
    natural_sort: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, object]]:
    """
    基于 target_pairs.csv 生成“按 lag 分组”的目标列表，并返回映射与诊断信息。

    Parameters
    ----------
    pairs : str | PathLike | DataFrame
        target_pairs.csv 的路径或已经读好的 DataFrame。至少包含列 [target_col, lag_col]。
    labels_df : DataFrame, optional
        训练标签表（train_labels.csv），若提供则只保留 labels 中实际存在的 target 列。
    target_col : str
        target 名称列名（一般是 'target'）。
    lag_col : str
        lag 列名（一般是 'lag'）。
    date_col : str
        日期列名（labels_df 中用于识别的日期列，默认 'date_id'）。
    allowed_lags : tuple[int]
        允许的 lag 集合（默认 (1,2,3,4)）。
    natural_sort : bool
        是否按 target_后缀的数字进行自然排序（若命名形如 'target_123'）。

    Returns
    -------
    lag_to_targets : dict[str, list[str]]
        形如 {'lag_1': ['target_0','target_4',...], 'lag_2': [...], ...}
    target_to_lag : dict[str, int]
        形如 {'target_0':1, 'target_1':4, ...}
    meta : dict
        诊断信息，包括：
        - 'all_pairs_targets' : set
        - 'label_targets'     : set | None
        - 'missing_from_labels': list   # pairs 有、labels 没有
        - 'extra_in_labels'     : list   # labels 有、pairs 没有
        - 'counts_by_lag'       : dict   # 每个 lag 下目标个数
    """
    # 读取/复制
    if isinstance(pairs, (str, os.PathLike)):
        pairs_df = pd.read_csv(pairs)
    else:
        pairs_df = pairs.copy()

    # 基本校验
    for col in (target_col, lag_col):
        if col not in pairs_df.columns:
            raise ValueError(f"`pairs` 缺少列: {col}")

    # 只保留必要列，去重，清洗
    pairs_df = pairs_df[[target_col, lag_col]].dropna().drop_duplicates()
    pairs_df[target_col] = pairs_df[target_col].astype(str).str.strip()
    # lag 转 int（非整数会报错，提早发现数据问题）
    pairs_df[lag_col] = pairs_df[lag_col].astype(int)

    # 记录 pairs 中的全部 target
    all_pairs_targets = set(pairs_df[target_col].unique())

    # 过滤允许的 lag
    if allowed_lags is not None:
        pairs_df = pairs_df[pairs_df[lag_col].isin(allowed_lags)]

    label_targets_set = None
    missing_from_labels = []
    extra_in_labels = []

    # 可选：与 labels 对齐，只保留 labels 中存在的 target 列
    if labels_df is not None:
        if date_col not in labels_df.columns:
            raise ValueError(f"`labels_df` 缺少日期列: {date_col}")
        label_targets_set = set(c for c in labels_df.columns if c != date_col)

        missing_from_labels = sorted(t for t in all_pairs_targets if t not in label_targets_set)
        extra_in_labels     = sorted(t for t in label_targets_set if t not in all_pairs_targets)

        # 只保留 labels 里有的 target
        pairs_df = pairs_df[pairs_df[target_col].isin(label_targets_set)]

    # 分组 & 排序
    def _nat_key(s: str):
        """natural sort: 'target_12' -> (prefix, 12)"""
        if not natural_sort:
            return s
        try:
            if "_" in s:
                prefix, suf = s.rsplit("_", 1)
                if suf.isdigit():
                    return (prefix, int(suf))
        except Exception:
            pass
        return (s, 0)

    lag_to_targets: Dict[str, List[str]] = {}
    target_to_lag: Dict[str, int] = {}

    for L, g in pairs_df.groupby(lag_col):
        tlist = sorted(g[target_col].unique().tolist(), key=_nat_key)
        lag_to_targets[f"lag_{int(L)}"] = tlist
        for t in tlist:
            target_to_lag[t] = int(L)

    counts_by_lag = {k: len(v) for k, v in lag_to_targets.items()}

    meta = {
        "all_pairs_targets": all_pairs_targets,
        "label_targets": label_targets_set,
        "missing_from_labels": missing_from_labels,
        "extra_in_labels": extra_in_labels,
        "counts_by_lag": counts_by_lag,
    }
    return lag_to_targets, target_to_lag, meta


    def _group_targets_by_lag(self, target_cols):
        """
        根据lag将目标分组
        """
        groups = {
            'lag1': [],
            'lag2': [],
            'lag3': [],
            'lag4': [],
            'other': []
        }

        for target in target_cols:
            if 'lag1' in target or target.endswith('_1'):
                groups['lag_1'].append(target)
            elif 'lag2' in target or target.endswith('_2'):
                groups['lag_2'].append(target)
            elif 'lag3' in target or target.endswith('_3'):
                groups['lag_3'].append(target)
            elif 'lag4' in target or target.endswith('_4'):
                groups['lag_4'].append(target)
            else:

                try:
                    target_num = int(target.split('_')[-1])
                    lag_group = (target_num % 4)+1
                    groups[f'lag{lag_group}'].append(target)
                except:
                    groups['other'].append(target)

        groups = {k: v for k, v in groups.items() if v}

        print(f"  ✓ 根据lag将目标分组: {groups}")
        return groups
    
    def create_time_fold(self, df, cv_folds=5):
        """
        创建基于date_id的严格时间折
        """
        print(f"创建{cv_folds}个严格时间折")

        unique_dates = sorted(df['date_id'].unique())
        n_dates = len(unique_dates)

        folds = []

        for fold in range(cv_folds):

            train_end_idx = int(n_dates * (0.6 + fold * 0.08)) 

            val_start_idx = train_end_idx
            val_end_idx = min(train_end_idx + int(n_dates * 0.1), n_dates)

            if val_start_idx >= n_dates or val_end_idx <= val_start_idx:
                continue

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            train_mask = df['date_id'].isin(train_dates)
            val_mask = df['date_id'].isin(val_dates)

            folds.append({
                'train_indices': df[train_mask].index.tolist(),
                'val_indices': df[val_mask].index.tolist(),
                'train_dates': train_dates,
                'val_dates': val_dates
            })

            print(f"    折{fold+1}: 训练({len(train_dates)}天) -> 验证({len(val_dates)}天)")
        
        return folds

    def _safe_impute_and_create_missing_indicators(self, df, feature_cols, fold_info):
        """
        安全的缺失值处理 + 创建缺失指示列 - 解决问题3
        """
        print("    执行安全的缺失值处理...")
        
        train_indices = fold_info['train_indices']
        
        # 在训练集上计算填充值
        train_data = df.loc[train_indices, feature_cols]
        
        # 使用训练集的中位数作为填充值
        fill_values = train_data.median()
        
        # 对于仍然是NaN的特征，用0填充
        fill_values = fill_values.fillna(0)
        
        # 创建缺失指示列
        missing_indicators = {}
        for col in feature_cols:
            if df[col].isna().any():
                missing_indicators[f'{col}_is_missing'] = df[col].isna().astype(int)
        
        # 填充缺失值（整个数据集都用训练集的统计量）
        df_filled = df.copy()
        df_filled[feature_cols] = df_filled[feature_cols].fillna(fill_values)
        
        # 添加缺失指示列
        for col_name, indicator in missing_indicators.items():
            df_filled[col_name] = indicator
        
        print(f"      创建了{len(missing_indicators)}个缺失指示列")
        print(f"      使用训练集统计量填充{len(feature_cols)}个特征")
        
        return df_filled, list(missing_indicators.keys())


def feature_selection_by_lag_cv(
    self,
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    max_features_per_lag: int = 50,
    cv_folds: int = 3,
    selection_method: str = 'daily_spearman_sharpe',
    date_col: str = 'date_id',
    filler: Union[float, int] = -999,
):
    """
    按 lag 分组进行特征选择（严格时间CV），解决：
      - 不同 target 的 lag 各不相同（按 lag 分组）
      - 时间因果（前训后验）
      - lag 对齐（用位置平移标签）
    支持：
      - 长表 df（含 'target' 列）：可以用 daily_spearman_sharpe
      - 宽表 df：退化为稳健 MI（无法做日内截面秩相关）
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.impute import SimpleImputer

    print(f"\n按 lag 分组的CV特征选择（method={selection_method}, folds={cv_folds}）...")

    # ---------------- helpers ----------------
    def _get_lag_from_group(name: str) -> int:
        # 允许 'lag_1' / 'L1' / 1
        if isinstance(name, (int, np.integer)): return int(name)
        s = str(name).lower().strip()
        for tok in ('lag_', 'l'):
            if s.startswith(tok):
                s = s[len(tok):]
                break
        return int(s)

    def _shift_labels_for_lag_position(y_wide: pd.DataFrame, lag: int) -> pd.DataFrame:
        """用'日期索引位置'进行平移：让 t 的特征 ↔ t+lag 的 y"""
        y = y_wide.copy().replace(filler, np.nan)
        uniq = np.array(sorted(pd.unique(y[date_col])))
        pos  = pd.Series(np.arange(len(uniq)), index=uniq)
        y = y[y[date_col].isin(pos.index)].copy()
        y['_pos'] = y[date_col].map(pos)
        y = y[y['_pos'] - lag >= 0].copy()
        y[date_col] = uniq[(y['_pos'] - lag).values]
        y = y.drop(columns=['_pos'])
        return y

    def _make_time_folds_by_date(dates: np.ndarray, n_folds: int):
        """简易时间折：按唯一日期顺序切成 n_folds 个 val 切片；train=之前所有日期"""
        uniq = np.array(sorted(pd.unique(dates)))
        n = len(uniq)
        fold_size = max(1, n // n_folds)
        folds = []
        for k in range(n_folds):
            va_start = k * fold_size
            va_end   = (k + 1) * fold_size if k < n_folds - 1 else n
            va_dates = uniq[va_start:va_end]
            tr_dates = uniq[:va_start]
            if len(tr_dates) == 0 or len(va_dates) == 0:
                continue
            folds.append({'train_dates': tr_dates, 'val_dates': va_dates})
        return folds

    def _daily_spearman_sharpe(y_true: pd.Series, y_pred: pd.Series, d: pd.Series) -> float:
        """按日分组的 Spearman → Sharpe（mean/std）"""
        g = (
            pd.DataFrame({'y': y_true, 'p': y_pred, 'd': d})
            .dropna()
            .groupby('d')
            .apply(lambda g_: g_['y'].rank().corr(g_['p'].rank(), method='spearman'))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(g) == 0: 
            return 0.0
        mu = g.mean()
        sd = g.std(ddof=0)
        return float(mu / (sd + 1e-12))

    # ---------------- prepare groups ----------------
    # 你自己的分组函数：返回 {'lag_1': [targets...], ...}
    target_cols = [c for c in labels_df.columns if c != date_col]
    target_groups = self._group_targets_by_lag(target_cols)  # 需已实现

    # 特征列
    feature_cols = [c for c in df.columns if c not in (date_col, 'target')]

    # 输出
    lag_results = {}

    # ---------------- per lag group ----------------
    for lag_group, group_targets in target_groups.items():
        lag = _get_lag_from_group(lag_group)
        print(f"\n处理 {lag_group}（lag={lag}，{len(group_targets)} 个目标）")

        # 1) 只保留本 lag 的标签列，并做 lag 对齐
        keep_targets = [t for t in group_targets if t in labels_df.columns]
        if not keep_targets:
            print("  * 跳过：该组在 labels 中无目标列")
            continue

        y_wide = labels_df[[date_col] + keep_targets].copy()
        y_shifted = _shift_labels_for_lag_position(y_wide, lag)

        # 2) 组装“监督表” data（长表/宽表分别处理）
        if 'target' in df.columns:
            # ---- 长表：按 (date,target) 精确对齐 ----
            feats = df[df['target'].isin(keep_targets)].copy()
            y_long = y_shifted.melt(id_vars=[date_col], var_name='target', value_name='y')
            y_long['target'] = y_long['target'].astype(str)
            data = feats.merge(y_long, on=[date_col, 'target'], how='inner')
        else:
            # ---- 宽表：只有 date 级特征，退化为 MI 路线 ----
            feats = df.copy()
            data = feats.merge(y_shifted, on=date_col, how='inner')

        # 安全检查
        if len(data) == 0:
            print("  * 跳过：对齐后无样本")
            continue

        # 3) 构造时间折（基于 data 的日期集合）
        folds = _make_time_folds_by_date(data[date_col].values, cv_folds)
        if not folds:
            print("  * 跳过：无法创建有效时间折")
            continue

        # 4) 逐折给每个特征打分
        cv_feature_scores = {f: [] for f in feature_cols}
        for fold_idx, fdict in enumerate(folds, 1):
            tr_dates = fdict['train_dates']; va_dates = fdict['val_dates']
            tr_mask = data[date_col].isin(tr_dates)
            va_mask = data[date_col].isin(va_dates)

            if tr_mask.sum() < 50 or va_mask.sum() < 10:
                print(f"  折 {fold_idx}: 样本不足，跳过")
                continue

            if 'target' in data.columns and selection_method == 'daily_spearman_sharpe':
                # ---------- 长表 + 日内 Sharpe ----------
                df_tr = data.loc[tr_mask, [date_col, 'target', 'y'] + feature_cols].copy()

                # 折内插补（特征），标签不插
                imputer = SimpleImputer(strategy='median')
                X_tr = pd.DataFrame(imputer.fit_transform(df_tr[feature_cols]),
                                    columns=feature_cols, index=df_tr.index)

                # 验证集（只 transform）
                df_va = data.loc[va_mask, [date_col, 'target', 'y'] + feature_cols].copy()
                X_va = pd.DataFrame(imputer.transform(df_va[feature_cols]),
                                    columns=feature_cols, index=df_va.index)

                # 单特征打分：把该特征当作“打分器”，看日内秩相关的 Sharpe
                for feat in feature_cols:
                    score = _daily_spearman_sharpe(
                        y_true=df_va['y'].values,
                        y_pred=X_va[feat].values,
                        d=df_va[date_col].values
                    )
                    cv_feature_scores[feat].append(score)

            else:
                # ---------- 宽表 或 指定方法不是日内 Sharpe：退化为“稳健 MI” ----------
                df_tr = data.loc[tr_mask, [date_col] + feature_cols + keep_targets].copy()

                # 折内插补（仅特征）
                imputer = SimpleImputer(strategy='median')
                X_tr = pd.DataFrame(imputer.fit_transform(df_tr[feature_cols]),
                                    columns=feature_cols, index=df_tr.index)

                # 对每个特征：与该 lag 组内各目标的 MI（只在该目标有 y 的样本）
                for feat in feature_cols:
                    feat_scores = []
                    x = X_tr[feat].values
                    valid_x = np.isfinite(x)
                    if valid_x.sum() < 50:
                        cv_feature_scores[feat].append(0.0)
                        continue
                    for tgt in keep_targets:
                        yv = df_tr[tgt].values
                        valid = valid_x & np.isfinite(yv)
                        if valid.sum() < 50:
                            continue
                        try:
                            mi = mutual_info_regression(
                                x[valid].reshape(-1,1), yv[valid], random_state=42, n_neighbors=3
                            )[0]
                            if np.isfinite(mi):
                                feat_scores.append(float(mi))
                        except Exception:
                            pass
                    if feat_scores:
                        # 稳健聚合：中位数 + 稳定性（小）惩罚
                        med = float(np.median(feat_scores))
                        mean = float(np.mean(feat_scores))
                        std = float(np.std(feat_scores))
                        robust = 0.7*med + 0.2*mean - 0.1*std
                        cv_feature_scores[feat].append(robust)
                    else:
                        cv_feature_scores[feat].append(0.0)

        # 5) 折间聚合 & 选 Top-K
        final_scores = {}
        for feat, scs in cv_feature_scores.items():
            if scs:
                med = float(np.median(scs))
                sd  = float(np.std(scs))
                # 稳定性惩罚：分数越不稳，扣得越多
                final_scores[feat] = med - 0.1 * (sd / (abs(med) + 1e-12))
            else:
                final_scores[feat] = 0.0

        ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [f for f,_ in ranked[:max_features_per_lag]]

        lag_results[lag_group] = {
            'selected_features': selected,
            'scores': final_scores,
        }
        print(f"  ✓ {lag_group}: 选择 {len(selected)}/{len(feature_cols)} 个特征")

    return lag_results




def _robust_mi_selection_cv(self, merged_df, feature_cols, target_cols, 
                           max_features, cv_folds, fold_size):
    """
    使用对所有目标的稳健性互信息进行特征选择
    """
    print("  使用稳健性互信息选择...")
    
    n = len(merged_df)
    cv_feature_scores = {feat: [] for feat in feature_cols}  # 存储每折的评分
    
    for fold in range(cv_folds):
        print(f"    处理第 {fold + 1}/{cv_folds} 折...")
        
        # 定义训练集（当前折之外的所有数据）
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n
        
        train_indices = list(range(0, val_start)) + list(range(val_end, n))
        
        train_data = merged_df.iloc[train_indices].copy()
        
        # 准备训练数据
        X_train = train_data[feature_cols].fillna(0)
        
        # 计算每个特征对所有目标的互信息
        feature_mi_scores = {}
        
        for feature in feature_cols:
            if feature not in X_train.columns:
                continue
                
            x_feature = X_train[feature].values
            mi_scores_all_targets = []
            
            # 对每个目标计算互信息
            for target in target_cols:
                if target not in train_data.columns:
                    continue
                    
                y_target = train_data[target].fillna(0).values
                
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi_score = mutual_info_regression(
                        x_feature.reshape(-1, 1), y_target, 
                        random_state=42
                    )[0]
                    
                    if not np.isnan(mi_score):
                        mi_scores_all_targets.append(mi_score)
                        
                except Exception as e:
                    continue
            
            # 计算稳健性评分（中位数 + 稳定性奖励）
            if mi_scores_all_targets:
                median_mi = np.median(mi_scores_all_targets)
                mean_mi = np.mean(mi_scores_all_targets)
                std_mi = np.std(mi_scores_all_targets)
                
                # 稳健性评分：偏向在多个目标上都表现稳定的特征
                robust_score = 0.7 * median_mi + 0.2 * mean_mi - 0.1 * std_mi
                feature_mi_scores[feature] = robust_score
            else:
                feature_mi_scores[feature] = 0.0
        
        # 存储该折的评分
        for feature, score in feature_mi_scores.items():
            cv_feature_scores[feature].append(score)
    
    # 计算最终的稳健性评分
    final_feature_scores = {}
    for feature, scores in cv_feature_scores.items():
        if scores:
            # 使用中位数作为最终评分（更稳健）
            final_score = np.median(scores)
            score_std = np.std(scores)
            
            # 惩罚在不同折上评分差异很大的特征
            stability_penalty = score_std / (abs(final_score) + 1e-8)
            final_feature_scores[feature] = final_score - 0.1 * stability_penalty
        else:
            final_feature_scores[feature] = 0.0
    
    # 选择top特征
    sorted_features = sorted(
        final_feature_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    print(f"  ✓ 选择了 {len(selected_features)} 个特征")
    print(f"  前5个特征评分: {sorted_features[:5]}")
    
    return selected_features, final_feature_scores


def _per_target_selection_cv(self, merged_df, feature_cols, target_cols, 
                            max_features, cv_folds, fold_size):
    """
    为每个目标分别进行特征选择，然后综合
    """
    print("  为每个目标分别选择特征...")
    
    n = len(merged_df)
    per_target_features = {}
    features_per_target = max(max_features // 10, 20)  # 每个目标选择的特征数
    
    # 目标分组（可根据业务逻辑调整）
    target_groups = self._group_targets(target_cols)
    
    for group_name, group_targets in target_groups.items():
        print(f"    处理目标组: {group_name} ({len(group_targets)}个目标)...")
        
        group_feature_scores = {feat: [] for feat in feature_cols}
        
        for fold in range(cv_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n
            train_indices = list(range(0, val_start)) + list(range(val_end, n))
            
            train_data = merged_df.iloc[train_indices].copy()
            X_train = train_data[feature_cols].fillna(0)
            
            # 为该组目标计算特征重要性
            for feature in feature_cols:
                if feature not in X_train.columns:
                    continue
                    
                x_feature = X_train[feature].values.reshape(-1, 1)
                feature_scores = []
                
                for target in group_targets:
                    if target not in train_data.columns:
                        continue
                        
                    y_target = train_data[target].fillna(0).values
                    
                    try:
                        from sklearn.feature_selection import mutual_info_regression
                        mi_score = mutual_info_regression(
                            x_feature, y_target, random_state=42
                        )[0]
                        
                        if not np.isnan(mi_score):
                            feature_scores.append(mi_score)
                    except:
                        continue
                
                # 计算该特征对该组目标的平均重要性
                if feature_scores:
                    group_feature_scores[feature].append(np.mean(feature_scores))
                else:
                    group_feature_scores[feature].append(0.0)
        
        # 计算该组的最终特征评分
        group_final_scores = {}
        for feature, scores in group_feature_scores.items():
            if scores:
                group_final_scores[feature] = np.median(scores)
            else:
                group_final_scores[feature] = 0.0
        
        # 为该组选择top特征
        sorted_group_features = sorted(
            group_final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        group_selected = [feat for feat, score in sorted_group_features[:features_per_target]]
        per_target_features[group_name] = group_selected
        
        print(f"      为{group_name}选择了{len(group_selected)}个特征")
    
    # 合并所有组选择的特征
    all_selected_features = set()
    for group_features in per_target_features.values():
        all_selected_features.update(group_features)
    
    final_selected = list(all_selected_features)[:max_features]
    
    print(f"  ✓ 总共选择了 {len(final_selected)} 个唯一特征")
    
    return final_selected, per_target_features


def _variance_weighted_selection_cv(self, merged_df, feature_cols, target_cols, 
                                   max_features, cv_folds, fold_size):
    """
    使用方差加权的特征选择
    """
    print("  使用方差加权特征选择...")
    
    n = len(merged_df)
    cv_feature_scores = {feat: [] for feat in feature_cols}
    
    for fold in range(cv_folds):
        print(f"    处理第 {fold + 1}/{cv_folds} 折...")
        
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n
        train_indices = list(range(0, val_start)) + list(range(val_end, n))
        
        train_data = merged_df.iloc[train_indices].copy()
        
        # 计算目标变量的方差（用作权重）
        target_variances = {}
        for target in target_cols:
            if target in train_data.columns:
                var = train_data[target].fillna(0).var()
                target_variances[target] = var if var > 0 else 1e-8
        
        # 标准化权重
        total_var = sum(target_variances.values())
        target_weights = {t: v/total_var for t, v in target_variances.items()}
        
        X_train = train_data[feature_cols].fillna(0)
        
        # 计算加权互信息
        for feature in feature_cols:
            if feature not in X_train.columns:
                continue
                
            x_feature = X_train[feature].values.reshape(-1, 1)
            weighted_mi = 0.0
            
            for target, weight in target_weights.items():
                if target not in train_data.columns:
                    continue
                    
                y_target = train_data[target].fillna(0).values
                
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi_score = mutual_info_regression(
                        x_feature, y_target, random_state=42
                    )[0]
                    
                    if not np.isnan(mi_score):
                        weighted_mi += weight * mi_score
                        
                except:
                    continue
            
            cv_feature_scores[feature].append(weighted_mi)
    
    # 计算最终评分
    final_feature_scores = {}
    for feature, scores in cv_feature_scores.items():
        if scores:
            final_feature_scores[feature] = np.mean(scores)
        else:
            final_feature_scores[feature] = 0.0
    
    # 选择top特征
    sorted_features = sorted(
        final_feature_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    print(f"  ✓ 选择了 {len(selected_features)} 个特征")
    
    return selected_features, final_feature_scores


def _group_targets(self, target_cols):
    """
    将424个目标分组（可根据业务逻辑自定义）
    """
    # 简单按编号分组，实际使用时可根据业务逻辑细化
    groups = {}
    group_size = 50  # 每组50个目标
    
    for i in range(0, len(target_cols), group_size):
        group_name = f'target_group_{i//group_size}'
        groups[group_name] = target_cols[i:i+group_size]
    
    return groups


def apply_feature_selection_cv(self, df_features, labels_df, 
                              selection_method='robust_mi', 
                              max_features=200, cv_folds=3):
    """
    应用特征选择的主函数
    """
    print("=" * 60)
    print("开始CV特征选择")
    print("=" * 60)
    
    # 检查特征数量
    feature_cols = [col for col in df_features.columns if col != 'date_id']
    original_feature_count = len(feature_cols)
    
    print(f"原始特征数: {original_feature_count}")
    
    if original_feature_count <= max_features:
        print(f"特征数已经少于{max_features}，无需选择")
        return df_features, {}
    
    # 执行特征选择
    selected_features, feature_scores = self.feature_selection_cv(
        df_features, labels_df,
        max_features=max_features,
        cv_folds=cv_folds,
        selection_method=selection_method
    )
    
    # 保留选中的特征
    final_cols = ['date_id'] + selected_features
    df_selected = df_features[final_cols].copy()
    
    # 保存特征选择结果
    selection_info = {
        'method': selection_method,
        'original_features': original_feature_count,
        'selected_features': len(selected_features),
        'selected_feature_names': selected_features,
        'feature_scores': feature_scores
    }
    
    # 保存到文件
    import pandas as pd
    selection_df = pd.DataFrame([
        {'feature': feat, 'score': score} 
        for feat, score in feature_scores.items() 
        if feat in selected_features
    ])
    selection_df = selection_df.sort_values('score', ascending=False)
    selection_df.to_csv('feature_selection_cv_results.csv', index=False)
    
    print(f"\n特征选择完成:")
    print(f"  原始特征: {original_feature_count}")
    print(f"  选择特征: {len(selected_features)}")
    print(f"  压缩比例: {len(selected_features)/original_feature_count:.2%}")
    print(f"  结果已保存: feature_selection_cv_results.csv")
    
    return df_selected, selection_info

    def create_all_features(self, df,
                            temporal=True,
                            cross_sectional=True,
                            market_regime=True,
                            pca=True,
                            interaction=True,
                            remove_correlated=True,
                            correlation_threshold=0.98):
        """
        创建所有特征的主函数
        
        Parameters:
        -----------
        remove_correlated : bool
            是否移除高度相关的特征
        correlation_threshold : float
            相关性阈值，超过此值的特征对将被处理
        """
        print("=" * 60)
        print("开始全面特征工程")
        print("=" * 60)

        df_result = df.copy()
        original_cols = len(df.columns)

        # 1. 时间特征
        if temporal:
            # 只对价格相关的列创建时间特征（避免特征爆炸）
            price_cols = [col for col in df.columns
                          if any(x in col for x in ['close', 'Close', 'open', 'Open'])
                          and not col.endswith('_is_active')][:30]  # é™åˆ¶æ•°é‡
            df_result = self.create_temporal_features(df_result, columns=price_cols)

        # 2. 横截面特征
        if cross_sectional:
            df_result = self.create_cross_sectional_features(df_result)

        # 3. 市场状态特征
        if market_regime:
            df_result = self.create_market_regime_features(df_result)

        # 4. PCA特征
        if pca:
            df_result = self.create_pca_features(df_result, n_components=20)

        # 5. 交互特征
        if interaction:
            df_result = self.create_interaction_features(df_result, max_interactions=30)

        # 6. 强共线性清理（在最终特征集上）
        if remove_correlated:
            df_result, removed_features = self.remove_highly_correlated_features(
                df_result, 
                correlation_threshold=correlation_threshold,
                verbose=True
            )

        print(f"\n特征工程完成!")
        print(f"原始特征数: {original_cols}")
        print(f"最终特征数: {len(df_result.columns)}")
        print(f"新增特征数: {len(df_result.columns) - original_cols}")
        if remove_correlated:
            print(f"共线性清理: 移除了 {len(removed_features)} 个高度相关特征")

        return df_result


# ===========================
# 完整的特征工程流程示例
# ===========================

def complete_feature_engineering_pipeline():
    """
    完整的特征工程流程，包含所有步骤
    """

    print("=" * 60)
    print("开始完整的特征工程流程")
    print("=" * 60)

    # Step 1: 读取已处理缺失值的数据
    print("\nStep 1: 读取数据...")

    # 这个df是你之前处理过缺失值的训练数据
    # 如果你已经运行过缺失值处理，应该有一个 'train_final.csv' 或类似的文件
    df = pd.read_csv('train_final.csv')  # 或者使用你保存的处理后数据文件名
    print(f"  数据维度: {df.shape}")
    print(f"  列数: {len(df.columns)}")

    # Step 2: 读取目标数据（用于后续特征选择）
    labels_df = pd.read_csv('train_labels.csv')
    print(f"  目标数据维度: {labels_df.shape}")

    # Step 3: 创建特征工程器实例
    print("\nStep 2: 初始化特征工程器...")

    # ç›´æŽ¥ä½¿ç”¨å·²ç»å®šä¹‰çš„ç±»ï¼Œä¸éœ€è¦å¯¼å…¥
    engineer = TimeSeriesFeatureEngineer(
        window_sizes=[5, 10, 20, 60],  # 滚动窗口大小
        target_cols=[col for col in labels_df.columns if col != 'date_id']
    )

    # Step 4: 逐步创建不同类型的特征

    # 4.1 时间特征（只对价格列，避免特征爆炸）
    print("\nStep 3: 创建时间特征...")
    price_cols = [col for col in df.columns
                  if any(x in col.lower() for x in ['close', 'open', 'high', 'low'])
                  and not col.endswith('_is_active')][:30]  # é™åˆ¶æ•°é‡

    df_with_temporal = engineer.create_temporal_features(df, columns=price_cols)
    print(f"  当前特征数: {len(df_with_temporal.columns)}")

    # 4.2 横截面特征（包括从target_pairs.csv读取的资产对）
    print("\nStep 4: 创建横截面特征...")
    df_with_cross = engineer.create_cross_sectional_features(
        df_with_temporal,
        pairs_file='target_pairs.csv'  # 使用target_pairs.csv中的资产对
    )
    print(f"  当前特征数: {len(df_with_cross.columns)}")

    # 4.3 市场状态特征
    print("\nStep 5: 创建市场状态特征...")
    df_with_market = engineer.create_market_regime_features(df_with_cross)
    print(f"  当前特征数: {len(df_with_market.columns)}")

    # 4.4 PCA特征（降维）
    print("\nStep 6: 创建PCA特征...")
    df_with_pca = engineer.create_pca_features(
        df_with_market,
        n_components=20  # 每组保留20个主成分
    )
    print(f"  当前特征数: {len(df_with_pca.columns)}")

    # 4.5 交互特征
    print("\nStep 7: 创建交互特征...")
    # é€‰æ‹©é‡è¦çš„ç‰¹å¾è¿›è¡Œäº¤äº’
    important_features = [
        'US_Stock_GLD_adj_close',
        'FX_USDJPY',
        'US_Stock_XLE_adj_close',
        'LME_AH_Close'
    ]
    # 过滤实际存在的特征
    important_features = [f for f in important_features if f in df_with_pca.columns]

    df_with_interactions = engineer.create_interaction_features(
        df_with_pca,
        important_features=important_features,
        max_interactions=30
    )
    print(f"  最终特征数: {len(df_with_interactions.columns)}")

    # Step 5: 特征选择（如果特征太多）
    if len(df_with_interactions.columns) > 300:  # 如果特征超过300个
        print(f"\nStep 8: 特征选择（从{len(df_with_interactions.columns)}个特征中选择200个）...")

        from sklearn.feature_selection import mutual_info_regression

        # 合并特征和目标
        merged_df = pd.merge(df_with_interactions, labels_df, on='date_id', how='inner')

        # 选择一个目标进行特征重要性评估
        feature_cols = [col for col in df_with_interactions.columns if col != 'date_id']
        target_col = 'target_0'  # 使用第一个目标

        X = merged_df[feature_cols].fillna(0)
        y = merged_df[target_col].fillna(0)

        # 计算互信息
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # 选择top 200特征
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        top_features = feature_importance.head(200)['feature'].tolist()

        # 保留选中的特征
        df_final = df_with_interactions[['date_id'] + top_features]

        # 保存特征重要性
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"  ç‰¹å¾é‡è¦æ€§å·²ä¿å­˜")
    else:
        df_final = df_with_interactions

    # Step 6: 保存最终特征
    print(f"\nStep 9: 保存最终特征...")
    df_final.to_csv('train_features_complete.csv', index=False)
    print(f"  ✓ 特征已保存至: train_features_complete.csv")
    print(f"  最终数据维度: {df_final.shape}")

    # Step 7: ç‰¹å¾ç»Ÿè®¡
    print("\n特征类型统计:")
    feature_stats = {
        'lag特征': len([c for c in df_final.columns if 'lag_' in c]),
        '移动平均': len([c for c in df_final.columns if '_ma_' in c]),
        '波动率': len([c for c in df_final.columns if '_std_' in c]),
        '收益率': len([c for c in df_final.columns if '_return_' in c]),
        '相关性': len([c for c in df_final.columns if '_corr_' in c]),
        'PCA特征': len([c for c in df_final.columns if '_PC' in c]),
        '交互特征': len([c for c in df_final.columns if '_x_' in c or '_div_' in c]),
        '市场状态': len([c for c in df_final.columns if 'regime' in c or 'volatility' in c]),
        '指示器': len([c for c in df_final.columns if '_is_active' in c])
    }

    for feat_type, count in feature_stats.items():
        if count > 0:
            print(f"  {feat_type}: {count}")

    return df_final


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # 运行快速版本
        df_features = complete_feature_engineering_pipeline()
    else:
        # 运行完整版本
        df_features = complete_feature_engineering_pipeline()

    print("\n✅ 特征工程完成！")
    print("下一步：使用生成的特征文件进行建模")