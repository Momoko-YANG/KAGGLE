import pandas as pd
import numpy as np


# 读取三张表格
def load_and_process_data(train_path='train.csv',
                          labels_path='train_labels.csv',
                          pairs_path='target_pairs.csv'):
    """
    读取训练数据、目标值和目标对应关系表

    Parameters:
    -----------
    train_path : str
        训练数据文件路径
    labels_path : str
        目标值文件路径
    pairs_path : str
        目标对应关系文件路径

    Returns:
    --------
    train_df : DataFrame
        训练数据
    labels_df : DataFrame
        带有实际含义列名的目标值数据
    target_mapping : dict
        target代号到实际含义的映射字典
    """

    # 1. 读取三张表
    print("正在读取数据文件...")
    train_df = pd.read_csv(train_path)
    labels_df = pd.read_csv(labels_path)
    pairs_df = pd.read_csv(pairs_path)

    print(f"训练数据维度: {train_df.shape}")
    print(f"目标值数据维度: {labels_df.shape}")
    print(f"目标对应关系数据维度: {pairs_df.shape}")

    # 2. 创建target代号到实际含义的映射
    # 根据pairs表创建映射字典
    target_mapping = {}

    # 为每个target创建唯一的标识
    # 假设pairs表中的'pair'列包含实际的含义描述
    for idx, row in pairs_df.iterrows():
        target_name = row['target']
        lag = row['lag']
        pair_name = row['pair']

        # 创建带lag信息的完整描述
        full_description = f"{pair_name}_lag{lag}"

        # target_0, target_1, ... 的命名规则
        target_col_name = f"target_{idx}"
        target_mapping[target_col_name] = full_description

    # 3. 重命名labels_df的列名（除了date_id）
    new_column_names = {'date_id': 'date_id'}  # 保留date_id不变

    for col in labels_df.columns:
        if col != 'date_id':
            if col in target_mapping:
                new_column_names[col] = target_mapping[col]
            else:
                # 如果映射中没有，保持原名
                new_column_names[col] = col

    # 应用新的列名
    labels_df_renamed = labels_df.rename(columns=new_column_names)

    # 4. 打印映射信息的示例
    print("\n目标列映射示例（前10个）：")
    for i, (old_name, new_name) in enumerate(list(new_column_names.items())[:11]):
        if old_name != 'date_id':
            print(f"  {old_name} -> {new_name}")
        if i >= 10:
            break

    return train_df, labels_df_renamed, target_mapping


# 合并训练数据和目标值
def merge_train_and_labels(train_df, labels_df):
    """
    将训练数据和目标值合并

    Parameters:
    -----------
    train_df : DataFrame
        训练数据
    labels_df : DataFrame
        目标值数据（已重命名）

    Returns:
    --------
    merged_df : DataFrame
        合并后的完整数据集
    """

    # 基于date_id进行合并
    merged_df = pd.merge(train_df, labels_df, on='date_id', how='inner')

    print(f"\n合并后数据维度: {merged_df.shape}")
    print(f"特征列数量: {len(train_df.columns) - 1}")  # 减去date_id
    print(f"目标列数量: {len(labels_df.columns) - 1}")  # 减去date_id

    return merged_df


# 数据探索和验证
def explore_data(train_df, labels_df, target_mapping):
    """
    探索数据的基本信息

    Parameters:
    -----------
    train_df : DataFrame
        训练数据
    labels_df : DataFrame
        目标值数据（已重命名）
    target_mapping : dict
        目标映射字典
    """

    print("\n=== 数据探索 ===")

    # 1. 检查缺失值
    print("\n训练数据缺失值统计:")
    train_missing = train_df.isnull().sum()
    if train_missing.sum() > 0:
        print(f"存在缺失值的列数: {(train_missing > 0).sum()}")
        print(f"总缺失值数量: {train_missing.sum()}")
    else:
        print("无缺失值")

    print("\n目标数据缺失值统计:")
    labels_missing = labels_df.isnull().sum()
    if labels_missing.sum() > 0:
        print(f"存在缺失值的列数: {(labels_missing > 0).sum()}")
        print(f"总缺失值数量: {labels_missing.sum()}")
    else:
        print("无缺失值")

    # 2. 数据类型分析
    print("\n特征数据类型分布:")
    feature_cols = [col for col in train_df.columns if col != 'date_id']

    # 按前缀分组统计
    feature_groups = {}
    for col in feature_cols:
        prefix = col.split('_')[0]
        if prefix not in feature_groups:
            feature_groups[prefix] = []
        feature_groups[prefix].append(col)

    print("特征组分布:")
    for group, cols in feature_groups.items():
        print(f"  {group}: {len(cols)} 个特征")

    # 3. 目标变量分析
    print("\n目标变量信息:")
    print(f"目标变量总数: {len(target_mapping)}")

    # 统计不同lag的数量
    lag_counts = {}
    for target_desc in target_mapping.values():
        if 'lag' in target_desc:
            lag = target_desc.split('lag')[-1]
            lag_counts[f'lag{lag}'] = lag_counts.get(f'lag{lag}', 0) + 1

    print("不同lag的目标变量数量:")
    for lag, count in sorted(lag_counts.items()):
        print(f"  {lag}: {count} 个")


# 主函数
def main():
    """
    主执行函数
    """

    # 加载数据
    train_df, labels_df_renamed, target_mapping = load_and_process_data()

    # 合并数据
    full_df = merge_train_and_labels(train_df, labels_df_renamed)

    # 数据探索
    explore_data(train_df, labels_df_renamed, target_mapping)

    # 保存处理后的数据（可选）
    print("\n是否保存处理后的数据？")
    # labels_df_renamed.to_csv('train_labels_renamed.csv', index=False)
    # full_df.to_csv('train_full.csv', index=False)

    # 保存映射关系（可选）
    # import json
    # with open('target_mapping.json', 'w') as f:
    #     json.dump(target_mapping, f, indent=2)

    return train_df, labels_df_renamed, target_mapping, full_df


# 使用示例
if __name__ == "__main__":
    # 执行主函数
    train_df, labels_df_renamed, target_mapping, full_df = main()

    # 可以继续进行特征工程
    print("\n数据加载完成，可以开始特征工程！")

    # 示例：查看某个特定的目标变量
    print("\n查看前5个目标变量的描述性统计:")
    target_cols = [col for col in labels_df_renamed.columns if col != 'date_id'][:5]
    print(labels_df_renamed[target_cols].describe())