#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report (2015-2019) Analysis
# 
# 本 Notebook 演示如何将 2015~2019 年的世界幸福报告数据合并，并从时间序列和国家对比等角度进行探索性分析。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 为保证图表在 Notebook 中显示

# ---
# ## 1. 数据读取与合并

# 假设你有以下 5 个 CSV 文件，并且列名已对齐或重命名成一致:
# '2015.csv', '2016.csv', '2017.csv', '2018.csv', '2019.csv'
# 每个文件至少包含以下列:
# ["Country or region", "Score", "GDP per capita", "Social support", 
#  "Healthy life expectancy", "Freedom to make life choices", 
#  "Generosity", "Perceptions of corruption"]

files = {
    2015: "/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week3_Summary_Statistics/data/2015.csv",
    2016: "/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week3_Summary_Statistics/data/2016.csv",
    2017: "/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week3_Summary_Statistics/data/2017.csv",
    2018: "/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week3_Summary_Statistics/data/2018.csv",
    2019: "/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week3_Summary_Statistics/data/2019.csv"
}

dfs = []
for year, path in files.items():
    df_year = pd.read_csv(path)
    
    # 如果某些列名略有差异，可以在此处进行重命名，如:
    # df_year.rename(columns={'Happiness Score': 'Score'}, inplace=True)
    
    df_year['year'] = year
    dfs.append(df_year)

# 合并为一个 DataFrame
df_all = pd.concat(dfs, ignore_index=True)

print("合并后数据大小:", df_all.shape)
df_all.head()


# ---
# ## 2. 数据清洗与初步查看

# 检查是否有缺失值或重复行
print("\n缺失值情况：")
print(df_all.isnull().sum())

print("\n重复行数：", df_all.duplicated().sum())

# 去除可能的重复
df_all.drop_duplicates(inplace=True)

# 简要统计
df_all.describe(include='all')


# ---
# ## 3. 时间序列分析：全球或整体趋势

# 3.1 计算每年平均 Score，查看随时间的变化
mean_score_by_year = df_all.groupby('year')['Score'].mean()
print("\n各年份的平均幸福度：\n", mean_score_by_year)

plt.figure(figsize=(8,5))
plt.plot(mean_score_by_year.index, mean_score_by_year.values, marker='o', color='royalblue')
plt.xlabel("Year")
plt.ylabel("Average Happiness Score")
plt.title("Global Average Happiness Score (2015-2019)")
plt.grid(True)
plt.show()

# 3.2 如果你想看某些特定国家的趋势，例如 Finland, Denmark, Norway
countries_of_interest = ["Finland", "Denmark", "Norway"]
df_subset = df_all[df_all["Country or region"].isin(countries_of_interest)]

plt.figure(figsize=(8,5))
for country in countries_of_interest:
    c_data = df_subset[df_subset["Country or region"] == country].sort_values("year")
    plt.plot(c_data["year"], c_data["Score"], marker='o', label=country)

plt.xlabel("Year")
plt.ylabel("Happiness Score")
plt.title("Happiness Score Trend for Selected Countries")
plt.legend()
plt.grid(True)
plt.show()


# ---
# ## 4. 国家对比：排名与变化

# 4.1 查看 2019 年各国的 Score 排名，选取前 10 与后 10
df_2019 = df_all[df_all["year"] == 2019].copy()
df_2019_sorted = df_2019.sort_values("Score", ascending=False)
top_10 = df_2019_sorted.head(10)
bottom_10 = df_2019_sorted.tail(10)

print("\n2019 年幸福度 Top 10 国家：")
print(top_10[["Country or region", "Score"]])

print("\n2019 年幸福度 Bottom 10 国家：")
print(bottom_10[["Country or region", "Score"]])

# 条形图：2019 年 Top 10
plt.figure(figsize=(8,5))
plt.barh(top_10["Country or region"], top_10["Score"], color='green')
plt.gca().invert_yaxis()  # 让最高的排在最上方
plt.xlabel("Happiness Score")
plt.title("Top 10 Countries in 2019 by Happiness Score")
plt.show()

# 条形图：2019 年 Bottom 10
plt.figure(figsize=(8,5))
plt.barh(bottom_10["Country or region"], bottom_10["Score"], color='red')
plt.gca().invert_yaxis()
plt.xlabel("Happiness Score")
plt.title("Bottom 10 Countries in 2019 by Happiness Score")
plt.show()


# ---
# ## 5. 指标间相关性分析

# 合并数据后，可以查看 GDP、Social support 等对 Score 的相关性
# 注意：如果不同年度的指标口径略有变化，相关性仅供参考
numeric_cols = ["Score", "GDP per capita", "Social support", 
                "Healthy life expectancy", "Freedom to make life choices",
                "Generosity", "Perceptions of corruption"]

corr = df_all[numeric_cols].corr()
print("\n数值变量相关系数：\n", corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap among Key Variables")
plt.show()


# ---
# ## 6. 进一步分析：年份与国家交互

# 6.1 如果想比较各国在 2015-2019 年 Score 的均值，可以 groupby 两列
country_year_group = df_all.groupby(["Country or region", "year"])["Score"].mean().reset_index()

# 例如，查看某一国家在 5 年内是否大幅上升/下降
# 这里以 Finland 为例
finland_data = country_year_group[country_year_group["Country or region"] == "Finland"]
print("\nFinland's Score from 2015 to 2019:\n", finland_data)

# 6.2 如果想看每个国家在 5 年间平均 Score（不分年份），可再 groupby 一次
country_mean_score = df_all.groupby("Country or region")["Score"].mean().sort_values(ascending=False)
print("\n2015~2019 平均 Score 排名：")
print(country_mean_score.head(10))


# ---
# ## 7. 小结与下一步

# - 我们合并了 2015~2019 年数据，观察了全球平均幸福度随时间的变化
# - 对比了 2019 年国家排名，选取 Top 10 和 Bottom 10 做可视化
# - 对数值变量进行了相关性分析，GDP 与 Score 的相关性通常较高
# - 可进一步做回归或分地区（大洲）分析，需要在数据中添加“Continent”列后按地区分组

print("\n分析完成。根据需要可继续进行回归分析、分地区对比或异常值检测等。")
