import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 如果之前为显示中文设置过字体，现在要让图表以英文为主，则保持以下两行注释状态即可
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 数据集路径（请确保路径正确）
data_path = '/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week6_classification_decisiontree/data/compas-scores-raw.csv'
df = pd.read_csv(data_path)

# 打印数据预览及列名，确认数据结构
print("数据预览：")
print(df.head())
print("\n列名：")
print(df.columns)

#############################################
# Task 1: 按 Ethnic_Code_Text 分组，计算 DecileScore 统计信息
#############################################

# 按种族分组
grouped_ethnic = df.groupby('Ethnic_Code_Text')

# 计算每组的均值、中位数、标准差、样本数量以及范围（最大值-最小值）
stats_ethnic = grouped_ethnic['DecileScore'].agg(['mean', 'median', 'std', 'count'])
stats_ethnic['range'] = grouped_ethnic['DecileScore'].max() - grouped_ethnic['DecileScore'].min()

# 计算 95% CI：1.96 * (std / sqrt(n))
z = 1.96
stats_ethnic['CI'] = z * (stats_ethnic['std'] / np.sqrt(stats_ethnic['count']))

print("\nDecileScore 统计数据（按 Ethnic_Code_Text 分组）：")
print(stats_ethnic)

# 绘制按种族分组的 DecileScore 均值及 95% CI
plt.figure(figsize=(10, 6))
plt.bar(stats_ethnic.index, stats_ethnic['mean'], yerr=stats_ethnic['CI'], capsize=5, color='skyblue')
# 将图表标签改为英文
plt.xlabel('Ethnicity (Ethnic_Code_Text)')
plt.ylabel('Mean DecileScore')
plt.title('Mean DecileScore by Ethnicity (with 95% CI)')
plt.show()

#############################################
# Task 2: 按 Sex_Code_Text 和 RecSupervisionLevelText 进行分组
#############################################

# 如果没有 RecSupervisionLevelText，则只按 Sex_Code_Text 分组
if 'RecSupervisionLevelText' in df.columns:
    group_cols = ['Sex_Code_Text', 'RecSupervisionLevelText']
else:
    group_cols = ['Sex_Code_Text']

grouped_cross = df.groupby(group_cols)
stats_cross = grouped_cross['DecileScore'].agg(['mean', 'std', 'count'])
stats_cross['CI'] = z * (stats_cross['std'] / np.sqrt(stats_cross['count']))

print("\nDecileScore 统计数据（按 Sex_Code_Text 和 RecSupervisionLevelText 分组）：")
print(stats_cross)

# 如果有两个分组列，则使用 unstack() 并绘图
if len(group_cols) > 1:
    pivot_mean = stats_cross['mean'].unstack(group_cols[1])
    pivot_CI = stats_cross['CI'].unstack(group_cols[1])
    
    pivot_mean.plot(kind='bar', yerr=pivot_CI, capsize=4, figsize=(10, 6))
    # 将图表标签改为英文
    plt.xlabel('Sex (Sex_Code_Text)')
    plt.ylabel('Mean DecileScore')
    plt.title('Mean DecileScore by Sex and Supervision Level (with 95% CI)')
    plt.legend(title=group_cols[1])
    plt.show()
else:
    # 只有一个分组列
    plt.figure(figsize=(10, 6))
    plt.bar(stats_cross.index, stats_cross['mean'], yerr=stats_cross['CI'], capsize=5, color='lightgreen')
    plt.xlabel('Sex (Sex_Code_Text)')
    plt.ylabel('Mean DecileScore')
    plt.title('Mean DecileScore by Sex (with 95% CI)')
    plt.show()

#############################################
# Task 4: 探讨样本量对 95% CI 的影响
#############################################

# 定义函数，计算平均值及 95% CI
def compute_CI(sample, col, z=1.96):
    mean_val = sample[col].mean()
    std_val = sample[col].std()
    n = sample.shape[0]
    error = z * (std_val / np.sqrt(n))
    return mean_val, error

sample_sizes = [30, 300]

print("\n样本量对 DecileScore 95% CI 的影响：")
for n in sample_sizes:
    sample = df.sample(n, random_state=42)
    mean_val, error = compute_CI(sample, 'DecileScore')
    print(f"样本量: {n}")
    print(f"DecileScore 均值: {mean_val:.2f}")
    print(f"95% CI: {mean_val:.2f} ± {error:.2f} (区间宽度: {error*2:.2f})")
    print("-" * 40)
