import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_high_missing_columns(df, threshold=0.5, exclude_cols=['date_id']):
    """
    分析高缺失率的列，识别可能停止交易的特征

    Parameters:
    -----------
    df : DataFrame
        数据框
    threshold : float
        缺失率阈值（默认50%）
    exclude_cols : list
        排除的列

    Returns:
    --------
    high_missing_info : DataFrame
        高缺失率列的详细信息
    """

    high_missing_info = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / len(df)

        if missing_rate >= threshold:
            # 找出最后一个非缺失值的位置
            non_null_mask = df[col].notna()
            if non_null_mask.any():
                last_valid_idx = df[non_null_mask].index[-1]
                first_valid_idx = df[non_null_mask].index[0]

                # 检查缺失值模式
                is_discontinued = check_if_discontinued(df[col])

                info = {
                    'column': col,
                    'missing_count': missing_count,
                    'missing_rate': missing_rate,
                    'first_valid_row': first_valid_idx,
                    'last_valid_row': last_valid_idx,
                    'valid_rows': len(df) - missing_count,
                    'is_likely_discontinued': is_discontinued
                }
            else:
                info = {
                    'column': col,
                    'missing_count': missing_count,
                    'missing_rate': missing_rate,
                    'first_valid_row': None,
                    'last_valid_row': None,
                    'valid_rows': 0,
                    'is_likely_discontinued': True
                }

            high_missing_info.append(info)

    return pd.DataFrame(high_missing_info)


def check_if_discontinued(series):
    """
    检查一个序列是否像是停止交易的模式
    （连续的非空值后跟着连续的空值）

    Parameters:
    -----------
    series : Series
        要检查的序列

    Returns:
    --------
    bool : 是否可能是停止交易
    """

    if series.isnull().all():
        return True

    # 找到最后一个非空值的位置
    last_valid_idx = series.last_valid_index()

    if last_valid_idx is None:
        return True

    # 检查最后一个有效值之后是否全是NaN
    after_last_valid = series.loc[last_valid_idx:].iloc[1:]

    if len(after_last_valid) > 0:
        # 如果之后全是NaN，很可能是停止交易
        if after_last_valid.isnull().all():
            return True

    return False


def handle_discontinued_features(df, missing_threshold=0.8, strategies='all'):
    """
    处理可能停止交易的特征

    Parameters:
    -----------
    df : DataFrame
        原始数据
    missing_threshold : float
        缺失率阈值，超过此值考虑特殊处理
    strategies : str or list
        处理策略：'drop', 'indicator', 'fill_zero', 'fill_last', 'keep_partial', 'all'

    Returns:
    --------
    dict : 不同策略处理后的数据框
    """

    df = df.copy()
    results = {}

    # 分析高缺失率列
    high_missing_info = analyze_high_missing_columns(df, threshold=missing_threshold)
    discontinued_cols = high_missing_info[high_missing_info['is_likely_discontinued'] == True]['column'].tolist()

    print(f"发现 {len(discontinued_cols)} 个可能停止交易的特征:")
    for col in discontinued_cols[:5]:  # 显示前5个
        print(f"  - {col}")
    if len(discontinued_cols) > 5:
        print(f"  ... 还有 {len(discontinued_cols) - 5} 个")

    if strategies == 'all':
        strategies = ['drop', 'indicator', 'fill_zero', 'fill_last', 'keep_partial']
    elif isinstance(strategies, str):
        strategies = [strategies]

    # 策略1: 直接删除
    if 'drop' in strategies:
        df_drop = df.drop(columns=discontinued_cols)
        results['drop'] = df_drop
        print(f"\n策略1 (删除): 移除 {len(discontinued_cols)} 列，剩余 {len(df_drop.columns)} 列")

    # 策略2: 添加停止交易指示器
    if 'indicator' in strategies:
        df_indicator = df.copy()
        for col in discontinued_cols:
            # 创建指示器列
            indicator_col = f"{col}_is_active"
            df_indicator[indicator_col] = df_indicator[col].notna().astype(int)

            # 填充原始列的缺失值
            df_indicator[col] = df_indicator[col].fillna(method='ffill').fillna(0)

        results['indicator'] = df_indicator
        print(f"\n策略2 (指示器): 添加 {len(discontinued_cols)} 个活跃指示器列")

    # 策略3: 填充为0（表示无交易）
    if 'fill_zero' in strategies:
        df_zero = df.copy()
        for col in discontinued_cols:
            df_zero[col] = df_zero[col].fillna(0)
        results['fill_zero'] = df_zero
        print(f"\n策略3 (填零): 将停止交易后的值填充为0")

    # 策略4: 使用最后有效值填充
    if 'fill_last' in strategies:
        df_last = df.copy()
        for col in discontinued_cols:
            df_last[col] = df_last[col].fillna(method='ffill').fillna(0)
        results['fill_last'] = df_last
        print(f"\n策略4 (前向填充): 使用最后交易价格填充")

    # 策略5: 只保留有效交易期间的数据
    if 'keep_partial' in strategies:
        df_partial = df.copy()

        # 找出所有停止交易特征的最早停止时间
        min_last_valid_row = float('inf')
        for _, row in high_missing_info.iterrows():
            if row['is_likely_discontinued'] and row['last_valid_row'] is not None:
                min_last_valid_row = min(min_last_valid_row, row['last_valid_row'])

        if min_last_valid_row != float('inf'):
            df_partial = df_partial.iloc[:min_last_valid_row + 1]
            print(f"\n策略5 (截断): 只保留前 {min_last_valid_row + 1} 行数据")

        results['keep_partial'] = df_partial

    return results, high_missing_info


def create_trading_features(df, discontinued_cols):
    """
    为停止交易的期货创建特殊特征

    Parameters:
    -----------
    df : DataFrame
        原始数据
    discontinued_cols : list
        停止交易的列名列表

    Returns:
    --------
    df_enhanced : DataFrame
        增强后的数据框
    """

    df_enhanced = df.copy()

    for col in discontinued_cols:
        # 1. 交易活跃度指标
        df_enhanced[f'{col}_is_trading'] = df[col].notna().astype(int)

        # 2. 距离最后交易日的天数
        last_valid_idx = df[col].last_valid_index()
        if last_valid_idx is not None:
            df_enhanced[f'{col}_days_since_last_trade'] = 0
            df_enhanced.loc[last_valid_idx + 1:, f'{col}_days_since_last_trade'] = range(1, len(df) - last_valid_idx)

        # 3. 交易期间的统计特征（只计算有效期间）
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            df_enhanced[f'{col}_historical_mean'] = valid_data.mean()
            df_enhanced[f'{col}_historical_std'] = valid_data.std()
            df_enhanced[f'{col}_historical_median'] = valid_data.median()

        # 4. 填充策略：使用历史均值或0
        df_enhanced[f'{col}_filled'] = df[col].fillna(valid_data.mean() if len(valid_data) > 0 else 0)

    return df_enhanced


def evaluate_strategies(df_original, results_dict, target_col=None):
    """
    评估不同策略的效果

    Parameters:
    -----------
    df_original : DataFrame
        原始数据
    results_dict : dict
        不同策略的结果
    target_col : str
        目标列名（如果有的话）

    Returns:
    --------
    evaluation : DataFrame
        评估结果
    """

    evaluation = []

    for strategy_name, df_processed in results_dict.items():
        eval_info = {
            'strategy': strategy_name,
            'n_features': len(df_processed.columns),
            'n_rows': len(df_processed),
            'total_missing': df_processed.isnull().sum().sum(),
            'missing_rate': df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1])
        }

        evaluation.append(eval_info)

    return pd.DataFrame(evaluation)


def visualize_discontinued_pattern(df, col_name, save_path=None):
    """
    可视化停止交易的模式

    Parameters:
    -----------
    df : DataFrame
        数据框
    col_name : str
        要可视化的列名
    save_path : str
        保存路径（可选）
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 子图1：原始数据和缺失值
    ax1 = axes[0]
    valid_mask = df[col_name].notna()
    ax1.plot(df.index[valid_mask], df[col_name][valid_mask], 'b-', label='Valid Data', alpha=0.7)
    ax1.axvline(x=df[valid_mask].index[-1] if valid_mask.any() else 0,
                color='r', linestyle='--', label='Last Trading Day')
    ax1.set_title(f'{col_name} - Trading Pattern')
    ax1.set_xlabel('Date Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：缺失值模式
    ax2 = axes[1]
    missing_pattern = df[col_name].isnull().astype(int)
    ax2.fill_between(df.index, 0, missing_pattern, alpha=0.5, color='red', label='Missing')
    ax2.fill_between(df.index, 0, 1 - missing_pattern, alpha=0.5, color='green', label='Valid')
    ax2.set_title(f'{col_name} - Missing Value Pattern')
    ax2.set_xlabel('Date Index')
    ax2.set_ylabel('Data Availability')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


# 主处理函数
def process_discontinued_futures(file_path='train.csv',
                                 missing_threshold=0.8,
                                 strategy='indicator',
                                 save_result=False):
    """
    主函数：处理包含停止交易期货的数据

    Parameters:
    -----------
    file_path : str
        数据文件路径
    missing_threshold : float
        缺失率阈值
    strategy : str
        处理策略
    save_result : bool
        是否保存结果

    Returns:
    --------
    df_processed : DataFrame
        处理后的数据
    analysis_report : dict
        分析报告
    """

    print("=== 处理停止交易的期货特征 ===\n")

    # 读取数据
    df = pd.read_csv(file_path)
    print(f"原始数据维度: {df.shape}")

    # 执行分析和处理
    results, high_missing_info = handle_discontinued_features(
        df,
        missing_threshold=missing_threshold,
        strategies=strategy
    )

    # 选择最终策略
    if strategy in results:
        df_processed = results[strategy]
    else:
        df_processed = list(results.values())[0]

    # 评估结果
    evaluation = evaluate_strategies(df, results)

    print("\n=== 策略评估结果 ===")
    print(evaluation.to_string(index=False))

    # 生成报告
    analysis_report = {
        'high_missing_columns': high_missing_info,
        'evaluation': evaluation,
        'selected_strategy': strategy,
        'final_shape': df_processed.shape
    }

    # 保存结果
    if save_result:
        output_path = f'train_processed_{strategy}.csv'
        df_processed.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存至: {output_path}")

    return df_processed, analysis_report


# 使用示例
if __name__ == "__main__":
    # 处理数据 - 使用指示器策略（推荐）
    df_processed, report = process_discontinued_futures(
        file_path='train.csv',
        missing_threshold=0.8,  # 缺失率超过80%的列
        strategy='indicator',  # 使用指示器策略
        save_result=True
    )

    print(f"\n最终数据维度: {df_processed.shape}")
    print(f"高缺失率列数: {len(report['high_missing_columns'])}")

    # 如果想查看特定列的模式
    high_missing_cols = report['high_missing_columns']
    if len(high_missing_cols) > 0:
        first_discontinued = high_missing_cols.iloc[0]['column']
        visualize_discontinued_pattern(pd.read_csv('train.csv'), first_discontinued)

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class ComprehensiveMissingValueHandler:
    """
    综合缺失值处理器：先处理停止交易的期货，再处理常规缺失值
    """

    def __init__(self, discontinued_threshold=0.8, regular_missing_threshold=0.05):
        """
        Parameters:
        -----------
        discontinued_threshold : float
            判定为停止交易的缺失率阈值（默认80%）
        regular_missing_threshold : float
            常规缺失值的阈值（默认5%）
        """
        self.discontinued_threshold = discontinued_threshold
        self.regular_missing_threshold = regular_missing_threshold
        self.discontinued_cols = []
        self.regular_missing_cols = []
        self.processing_report = {}

    def analyze_missing_patterns(self, df, exclude_cols=['date_id']):
        """
        分析缺失值模式，区分停止交易和常规缺失
        """
        missing_analysis = []

        for col in df.columns:
            if col in exclude_cols:
                continue

            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / len(df)

            if missing_count > 0:
                # 检查是否是停止交易模式
                is_discontinued = self._check_discontinued_pattern(df[col])

                # 分类缺失值类型
                if missing_rate >= self.discontinued_threshold and is_discontinued:
                    missing_type = 'discontinued'
                    self.discontinued_cols.append(col)
                elif missing_count > 0:
                    missing_type = 'regular'
                    self.regular_missing_cols.append(col)
                else:
                    missing_type = 'none'

                missing_analysis.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_rate': missing_rate,
                    'missing_type': missing_type,
                    'last_valid_index': df[col].last_valid_index()
                })

        return pd.DataFrame(missing_analysis)

    def _check_discontinued_pattern(self, series):
        """
        检查是否为停止交易模式（连续有效值后跟连续缺失值）
        """
        if series.isnull().all():
            return True

        last_valid_idx = series.last_valid_index()
        if last_valid_idx is None:
            return True

        # 检查最后有效值之后是否全是缺失
        pos = series.index.get_loc(last_valid_idx)
        if pos < len(series) - 1:
            after_last = series.iloc[pos + 1:]
            if after_last.isnull().all() and len(after_last) > len(series) * 0.1:
                return True

        return False

    def process_all_missing_values(self, df,
                                   discontinued_strategy='indicator',
                                   regular_strategy='smart_fill'):
        """
        综合处理所有缺失值

        Parameters:
        -----------
        df : DataFrame
            原始数据
        discontinued_strategy : str
            停止交易列的处理策略 ('indicator', 'drop', 'fill_zero', 'fill_last')
        regular_strategy : str
            常规缺失值的处理策略 ('smart_fill', 'interpolate', 'mean', 'median')

        Returns:
        --------
        df_processed : DataFrame
            处理后的数据
        """
        print("=== 综合缺失值处理流程 ===\n")

        df_processed = df.copy()

        # Step 1: 分析缺失值模式
        print("Step 1: 分析缺失值模式...")
        missing_analysis = self.analyze_missing_patterns(df_processed)

        print(f"  - 停止交易的列: {len(self.discontinued_cols)} 个")
        print(f"  - 常规缺失的列: {len(self.regular_missing_cols)} 个")

        # Step 2: 处理停止交易的期货
        if self.discontinued_cols:
            print(f"\nStep 2: 处理停止交易的期货 (策略: {discontinued_strategy})...")
            df_processed = self._handle_discontinued_features(
                df_processed,
                self.discontinued_cols,
                discontinued_strategy
            )
            print(f"  ✓ 已处理 {len(self.discontinued_cols)} 个停止交易的特征")

        # Step 3: 处理常规缺失值
        if self.regular_missing_cols:
            print(f"\nStep 3: 处理常规缺失值 (策略: {regular_strategy})...")
            df_processed = self._handle_regular_missing(
                df_processed,
                self.regular_missing_cols,
                regular_strategy
            )
            print(f"  ✓ 已处理 {len(self.regular_missing_cols)} 个常规缺失特征")

        # Step 4: 最终验证
        print("\nStep 4: 最终验证...")
        remaining_missing = df_processed.isnull().sum().sum()
        print(f"  处理前总缺失值: {df.isnull().sum().sum()}")
        print(f"  处理后总缺失值: {remaining_missing}")

        if remaining_missing == 0:
            print("  ✓ 所有缺失值已成功处理！")
        else:
            print(f"  ⚠ 仍有 {remaining_missing} 个缺失值")

        # 生成处理报告
        self.processing_report = {
            'missing_analysis': missing_analysis,
            'discontinued_cols': self.discontinued_cols,
            'regular_missing_cols': self.regular_missing_cols,
            'discontinued_strategy': discontinued_strategy,
            'regular_strategy': regular_strategy,
            'original_missing': df.isnull().sum().sum(),
            'final_missing': remaining_missing
        }

        return df_processed

    def _handle_discontinued_features(self, df, cols, strategy):
        """
        处理停止交易的特征
        """
        df_result = df.copy()

        if strategy == 'indicator':
            # 添加交易状态指示器并填充
            for col in cols:
                # 创建指示器
                df_result[f'{col}_is_active'] = df_result[col].notna().astype(int)

                # 对于原始列，使用最后有效值填充（如果有的话）
                if df_result[col].notna().any():
                    last_valid_value = df_result[col].fillna(method='ffill').iloc[-1]
                    if pd.notna(last_valid_value):
                        df_result[col] = df_result[col].fillna(method='ffill')
                    else:
                        df_result[col] = df_result[col].fillna(0)
                else:
                    df_result[col] = 0

        elif strategy == 'drop':
            # 删除这些列
            df_result = df_result.drop(columns=cols)

        elif strategy == 'fill_zero':
            # 填充为0
            for col in cols:
                df_result[col] = df_result[col].fillna(0)

        elif strategy == 'fill_last':
            # 使用最后有效值填充
            for col in cols:
                df_result[col] = df_result[col].fillna(method='ffill').fillna(0)

        return df_result

    def _handle_regular_missing(self, df, cols, strategy):
        """
        处理常规缺失值（非停止交易）
        """
        df_result = df.copy()

        # 排除已经添加的指示器列
        cols_to_process = [col for col in cols if '_is_active' not in col]

        if strategy == 'smart_fill':
            # 智能填充：优先向前，首行向后
            for col in cols_to_process:
                if col in df_result.columns:  # 检查列是否存在（可能被删除）
                    # 先向前填充
                    df_result[col] = df_result[col].fillna(method='ffill')
                    # 剩余的（主要是首行）向后填充
                    df_result[col] = df_result[col].fillna(method='bfill')
                    # 如果还有缺失（整列都是NaN的情况），填充0
                    df_result[col] = df_result[col].fillna(0)

        elif strategy == 'interpolate':
            # 线性插值
            for col in cols_to_process:
                if col in df_result.columns:
                    df_result[col] = df_result[col].interpolate(method='linear', limit_direction='both')
                    df_result[col] = df_result[col].fillna(method='bfill').fillna(method='ffill').fillna(0)

        elif strategy == 'mean':
            # 均值填充
            for col in cols_to_process:
                if col in df_result.columns:
                    mean_value = df_result[col].mean()
                    if pd.notna(mean_value):
                        df_result[col] = df_result[col].fillna(mean_value)
                    else:
                        df_result[col] = df_result[col].fillna(0)

        elif strategy == 'median':
            # 中位数填充
            for col in cols_to_process:
                if col in df_result.columns:
                    median_value = df_result[col].median()
                    if pd.notna(median_value):
                        df_result[col] = df_result[col].fillna(median_value)
                    else:
                        df_result[col] = df_result[col].fillna(0)

        return df_result

    def get_processing_summary(self):
        """
        获取处理摘要
        """
        if not self.processing_report:
            return "尚未执行处理"

        summary = f"""
        ====== 缺失值处理摘要 ======

        原始缺失值总数: {self.processing_report['original_missing']}
        最终缺失值总数: {self.processing_report['final_missing']}

        停止交易的特征: {len(self.discontinued_cols)} 个
        处理策略: {self.processing_report['discontinued_strategy']}

        常规缺失的特征: {len(self.regular_missing_cols)} 个
        处理策略: {self.processing_report['regular_strategy']}

        处理成功率: {(1 - self.processing_report['final_missing'] / max(self.processing_report['original_missing'], 1)) * 100:.2f}%
        """

        return summary


def integrated_missing_value_pipeline(train_path='train.csv',
                                      save_path='train_processed.csv',
                                      discontinued_threshold=0.8,
                                      discontinued_strategy='indicator',
                                      regular_strategy='smart_fill',
                                      save_result=True):
    """
    集成的缺失值处理管道

    Parameters:
    -----------
    train_path : str
        训练数据路径
    save_path : str
        保存路径
    discontinued_threshold : float
        停止交易判定阈值
    discontinued_strategy : str
        停止交易处理策略
    regular_strategy : str
        常规缺失处理策略
    save_result : bool
        是否保存结果

    Returns:
    --------
    df_processed : DataFrame
        处理后的数据
    handler : ComprehensiveMissingValueHandler
        处理器对象（包含详细报告）
    """

    print("=" * 50)
    print("开始综合缺失值处理流程")
    print("=" * 50 + "\n")

    # 读取数据
    print("读取数据...")
    df = pd.read_csv(train_path)
    print(f"原始数据维度: {df.shape}\n")

    # 创建处理器
    handler = ComprehensiveMissingValueHandler(
        discontinued_threshold=discontinued_threshold
    )

    # 执行处理
    df_processed = handler.process_all_missing_values(
        df,
        discontinued_strategy=discontinued_strategy,
        regular_strategy=regular_strategy
    )

    # 打印摘要
    print(handler.get_processing_summary())

    # 保存结果
    if save_result:
        df_processed.to_csv(save_path, index=False)
        print(f"\n✓ 处理后的数据已保存至: {save_path}")

        # 保存处理报告
        report_path = save_path.replace('.csv', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(handler.get_processing_summary())
            f.write("\n\n=== 详细信息 ===\n")
            f.write(f"停止交易的列:\n{handler.discontinued_cols}\n\n")
            f.write(f"常规缺失的列:\n{handler.regular_missing_cols}\n")
        print(f"✓ 处理报告已保存至: {report_path}")

    return df_processed, handler


# 使用示例
if __name__ == "__main__":

    # 方案1：推荐配置
    print("执行推荐的处理方案...\n")

    df_processed, handler = integrated_missing_value_pipeline(
        train_path='train.csv',
        save_path='train_final.csv',
        discontinued_threshold=0.8,  # 缺失率>80%视为停止交易
        discontinued_strategy='indicator',  # 使用指示器策略处理停止交易
        regular_strategy='smart_fill',  # 智能填充常规缺失值
        save_result=True
    )

    print(f"\n最终数据维度: {df_processed.shape}")
    print(f"数据处理完成！")

    # 查看具体哪些列被识别为停止交易
    if handler.discontinued_cols:
        print(f"\n停止交易的特征（前5个）:")
        for col in handler.discontinued_cols[:5]:
            print(f"  - {col}")

    # 验证是否还有缺失值
    final_missing = df_processed.isnull().sum().sum()
    if final_missing == 0:
        print("\n✅ 完美！所有缺失值已被处理")
    else:
        print(f"\n⚠️ 注意：仍有 {final_missing} 个缺失值未处理")