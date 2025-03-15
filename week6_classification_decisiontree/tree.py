import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def print_tree_details(clf, feature_names, node_index=0, depth=0):
    """
    递归打印决策树每个节点的详细信息:
    - 特征名称 (feature)
    - 阈值 (threshold)
    - Gini impurity
    - 样本数 (n_node_samples)
    - value(各类别样本数)
    """
    left_child = clf.tree_.children_left[node_index]
    right_child = clf.tree_.children_right[node_index]
    threshold = clf.tree_.threshold[node_index]
    feature = clf.tree_.feature[node_index]
    impurity = clf.tree_.impurity[node_index]
    n_samples = clf.tree_.n_node_samples[node_index]
    value = clf.tree_.value[node_index]

    indent = "  " * depth  # 用于控制缩进，表示树的层次

    # 判断是否为叶子节点: 若 children_left[node_index] == -1 且 children_right[node_index] == -1
    if left_child == -1 and right_child == -1:
        # 叶子节点
        print(f"{indent}Leaf node {node_index}:")
        print(f"{indent}  gini = {impurity:.3f}, samples = {n_samples}, value = {value}")
    else:
        # 内部节点
        print(f"{indent}Node {node_index}:")
        print(f"{indent}  If {feature_names[feature]} <= {threshold:.3f} "
              f"(gini = {impurity:.3f}, samples = {n_samples}, value = {value}):")

        # 递归打印左子树
        print_tree_details(clf, feature_names, left_child, depth + 1)
        # 递归打印右子树
        print_tree_details(clf, feature_names, right_child, depth + 1)



# 1. 读取数据
df = pd.read_csv("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week6_classification_decisiontree/data/compas-scores-raw.csv")
print(df.columns)  # 确认实际列名

# 2. 数据清洗 & 特征工程
# 例如，假设目标变量为 "ScoreText"（需要根据你的实际任务确定目标）
# 同时，我们计算年龄（假设 DateOfBirth 格式为 YYYY-MM-DD），这里简单计算出生年份：
df["Age"] = pd.to_datetime(df["DateOfBirth"]).dt.year
current_year = pd.Timestamp.now().year
df["Age"] = current_year - df["Age"]

# 假设我们选择 Age, Sex_Code_Text, Ethnic_Code_Text, RawScore 作为特征
X = df[["Age", "Sex_Code_Text", "Ethnic_Code_Text"]]


# 假设目标变量是 ScoreText，需要先转换为数字编码（这里仅为示例）
# 你可以根据实际情况定义目标变量，这里先用 pd.factorize 进行编码
y, _ = pd.factorize(df["ScoreText"])

# 对类别型特征进行独热编码
X = pd.get_dummies(X, columns=["Sex_Code_Text", "Ethnic_Code_Text"], drop_first=True)

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 训练模型 (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

# 5. 训练模型 (Decision Tree)
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 5. 训练模型 (Decision Tree)
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

# 调用辅助函数，打印树的结构
print("\n===== Decision Tree Structure =====")
feature_names = X.columns  # 这里 X.columns 就是你原先构造的 DataFrame 的列名
print_tree_details(dt, feature_names.tolist())

# 然后再继续下面的预测和评估
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# 6. 可视化决策树（可选）
plt.figure(figsize=(15, 8), dpi=200)
plot_tree(dt, feature_names=X.columns, class_names=["Class_"+str(i) for i in np.unique(y)], filled=True)
plt.show()



print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# 6. 可视化决策树
plt.figure(figsize=(15, 8), dpi=200)
plot_tree(dt, feature_names=X.columns, class_names=["Class_"+str(i) for i in np.unique(y)], filled=True)
plt.show()
