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

    indent = "  " * depth

    if left_child == -1 and right_child == -1:
        # 叶子节点
        print(f"{indent}Leaf node {node_index}:")
        print(f"{indent}  gini = {impurity:.3f}, samples = {n_samples}, value = {value}")
    else:
        # 内部节点
        print(f"{indent}Node {node_index}:")
        print(f"{indent}  If {feature_names[feature]} <= {threshold:.3f} "
              f"(gini = {impurity:.3f}, samples = {n_samples}, value = {value}):")
        print_tree_details(clf, feature_names, left_child, depth + 1)
        print_tree_details(clf, feature_names, right_child, depth + 1)


# =========== 1. 读取数据 ===========
df = pd.read_csv("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week6_classification_decisiontree/data/compas-scores-raw.csv")
print(df.columns)  # 确认实际列名

# =========== 2. 数据清洗 & 特征工程 ===========
df["Age"] = pd.to_datetime(df["DateOfBirth"]).dt.year
current_year = pd.Timestamp.now().year
df["Age"] = current_year - df["Age"]

# 构造特征 X，包含 Age, RawScore, Sex_Code_Text, Ethnic_Code_Text
# 注意：RawScore 如果是数值，就不需要对它做 get_dummies
X = df[["Age", "RawScore", "Sex_Code_Text", "Ethnic_Code_Text"]]

# 目标变量
y, _ = pd.factorize(df["ScoreText"])

# 对 Sex_Code_Text, Ethnic_Code_Text 做独热编码 (drop_first=True)
X = pd.get_dummies(X, columns=["Sex_Code_Text", "Ethnic_Code_Text"], drop_first=True)

# =========== 3. 缩放敏感特征 ===========
# 识别敏感列: 包含 "Sex_Code_Text_" 或 "Ethnic_Code_Text_"
sensitive_cols = [col for col in X.columns if col.startswith("Sex_Code_Text_") or col.startswith("Ethnic_Code_Text_")]

# 将这些敏感列乘以 0.5 (你可根据需要调整这个系数)
for col in sensitive_cols:
    X[col] = X[col] * 0.5

# =========== 4. 标准化数值特征 ===========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========== 5. 划分训练集和测试集 ===========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# =========== 6. 训练模型 (KNN) 可选 ===========
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

# =========== 7. 训练模型 (Decision Tree) ===========
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 打印结构
print("\n===== Decision Tree Structure (RawScore + Scaled Sensitive) =====")
feature_names = X.columns.tolist()
print_tree_details(dt, feature_names)

# 查看评估指标
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

# =========== 8. 可视化决策树 (可选) ===========
plt.figure(figsize=(15, 8), dpi=200)
plot_tree(dt, feature_names=feature_names, class_names=[f"Class_{c}" for c in np.unique(y)], filled=True)
plt.show()
