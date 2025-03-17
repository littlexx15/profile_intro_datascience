import numpy as np
import matplotlib.pyplot as plt

# 定义平台名称及对应的数据类型数量
labels = ['Google', 'Instagram', 'Facebook']
values = [20, 11, 12]

# 雷达图需要闭合，所以重复第一个数据点
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

# 创建雷达图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 设置各个轴的标签
plt.xticks(angles[:-1], labels, fontsize=12)

# 设置径向标签（Y轴）
ax.set_rlabel_position(30)
plt.yticks([5, 10, 15, 20, 25], ["5", "10", "15", "20", "25"], color="grey", size=10)
plt.ylim(0, 25)

# 绘制雷达图线条并填充颜色
ax.plot(angles, values, color="red", linewidth=2, linestyle='solid')
ax.fill(angles, values, color="red", alpha=0.25)

# 添加标题
plt.title("各平台数据类型丰富度雷达图", size=15, y=1.1)

plt.show()
