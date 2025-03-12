import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# 用于交互式仪表盘
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

###############################################
# 1. 数据加载与预处理
###############################################

# 读取数据（确保文件在当前目录下）
data = pd.read_csv("/Users/xiangxiaoxin/Documents/GitHub/profile_intro_datascience/week7/data/propublica_data_for_fairml.csv")

print(data.columns)

# 查看数据基本信息
print(data.head())
print(data.columns)

# 目标变量：两年内是否再犯（1：再犯，0：不再犯）
y = data["Two_yr_Recidivism"]

# 特征：去掉目标列
X = data.drop(columns=["Two_yr_Recidivism"])

# 敏感属性：我们以 African_Am 为例（1：非裔；0：非非裔）
sensitive_name = "African_American"  # 改成实际列名
sensitive_features = X[sensitive_name]


# 划分训练集和测试集（保持目标分布平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

###############################################
# 2. 模型训练与预测
###############################################

# (1) 基线模型：逻辑回归
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
y_pred_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]

acc_baseline = accuracy_score(y_test, y_pred_baseline)
auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
eod_baseline = equalized_odds_difference(y_test, y_pred_baseline, sensitive_features=X_test[sensitive_name])

# 计算分组指标（基线模型）
metrics = {
    "accuracy": accuracy_score,
    "fpr": false_positive_rate,
    "fnr": false_negative_rate
}
metric_frame_baseline = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=X_test[sensitive_name]
)

# (2) 后处理模型：ThresholdOptimizer
post_model = ThresholdOptimizer(
    estimator=baseline_model,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
    predict_method="predict_proba"
)
post_model.fit(X_train, y_train, sensitive_features=X_train[sensitive_name])
# 得到后处理模型的预测结果
y_pred_post = post_model.predict(X_test, sensitive_features=X_test[sensitive_name])
# 注意：后处理模型内部采用概率预测，输出类别结果
# 为方便 ROC 曲线绘制，尝试调用内部概率方法（如果可用）
try:
    y_pred_proba_post = post_model._pmf_predict(X_test, sensitive_features=X_test[sensitive_name])
    # 如果返回的是二维数组，则取第二列
    if y_pred_proba_post.ndim == 2:
        y_pred_proba_post = y_pred_proba_post[:, 1]
except Exception as e:
    print("后处理模型预测概率出错：", e)
    y_pred_proba_post = np.zeros_like(y_pred_baseline)

acc_post = accuracy_score(y_test, y_pred_post)
auc_post = roc_auc_score(y_test, y_pred_proba_post) if np.any(y_pred_proba_post) else np.nan
eod_post = equalized_odds_difference(y_test, y_pred_post, sensitive_features=X_test[sensitive_name])
metric_frame_post = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_post,
    sensitive_features=X_test[sensitive_name]
)

# (3) 训练时公平约束模型：ExponentiatedGradient
exp_model = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000),
    constraints=EqualizedOdds()
)

exp_model.fit(X_train, y_train, sensitive_features=X_train[sensitive_name])
y_pred_exp = exp_model.predict(X_test)
# 对于概率，我们这里仅用类别预测计算 AUC（不理想，但展示用）
acc_exp = accuracy_score(y_test, y_pred_exp)
# 若能获取概率，则可以计算 AUC，但这里暂时仅展示类别
auc_exp = roc_auc_score(y_test, y_pred_exp)
eod_exp = equalized_odds_difference(y_test, y_pred_exp, sensitive_features=X_test[sensitive_name])
metric_frame_exp = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_exp,
    sensitive_features=X_test[sensitive_name]
)

###############################################
# 3. 整理各模型结果，便于后续绘图对比
###############################################

results_df = pd.DataFrame({
    "Model": ["Baseline", "Postprocessing", "ExpGradient"],
    "Accuracy": [acc_baseline, acc_post, acc_exp],
    "AUC": [auc_baseline, auc_post, auc_exp],
    "Equalized_Odds": [eod_baseline, eod_post, eod_exp],
    "Error_Rate": [1 - acc_baseline, 1 - acc_post, 1 - acc_exp]
})
print(results_df)

###############################################
# 4. 绘制各类图表（使用 Plotly）
###############################################

# (A) 分布图：展示 Two_yr_Recidivism 在不同 African_Am 群体中的分布（用直方图+密度图）
fig_dist = px.histogram(
    data, x="Two_yr_Recidivism", color=sensitive_name, barmode="overlay",
    histnorm='density', opacity=0.6,
    title="两年内再犯分布（按非裔 vs 非非裔）"
)
fig_dist.update_layout(xaxis_title="Two_yr_Recidivism", yaxis_title="Density")

# (B) 柱状图：展示基线模型下，不同群体的准确率、FPR 和 FNR
group_metrics = metric_frame_baseline.by_group.reset_index()
fig_bar = go.Figure()
for metric in ["accuracy", "fpr", "fnr"]:
    fig_bar.add_trace(go.Bar(
        x=group_metrics[sensitive_name].astype(str),
        y=group_metrics[metric],
        name=metric
    ))
fig_bar.update_layout(
    barmode="group",
    title="基线模型：不同群体的性能指标",
    xaxis_title=f"{sensitive_name} (0=非, 1=是)",
    yaxis_title="指标值"
)

# (C) ROC 曲线图：展示基线模型和后处理模型的 ROC 曲线
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
fpr_post, tpr_post, _ = roc_curve(y_test, y_pred_proba_post) if np.any(y_pred_proba_post) else ([], [], [])

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_baseline, y=tpr_baseline, mode='lines',
                             name=f"Baseline (AUC={auc_baseline:.2f})"))
if len(fpr_post) > 0:
    fig_roc.add_trace(go.Scatter(x=fpr_post, y=tpr_post, mode='lines',
                                 name=f"Postprocessing (AUC={auc_post:.2f})"))
fig_roc.update_layout(
    title="ROC 曲线对比",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)

# (D) 误差条图：这里以基线模型中各组 FPR 为例（示例中误差条使用模拟数据）
# 注意：实际中你可能需要利用 Bootstrap 或公式计算置信区间
group_fpr = metric_frame_baseline.by_group["fpr"].reset_index()
# 模拟误差（这里只是示例，真实情况需要具体计算）
group_fpr["fpr_err"] = 0.02

fig_error = go.Figure()
fig_error.add_trace(go.Bar(
    x=group_fpr[sensitive_name].astype(str),
    y=group_fpr["fpr"],
    error_y=dict(type='data', array=group_fpr["fpr_err"]),
    name="FPR with Error Bars"
))
fig_error.update_layout(
    title="各群体假正率（带误差条）",
    xaxis_title=f"{sensitive_name} (0=非, 1=是)",
    yaxis_title="False Positive Rate"
)

# (E) 散点图（性能-公平性权衡）：横轴为错误率，纵轴为 Equalized Odds 差异
fig_scatter = px.scatter(
    results_df, x="Error_Rate", y="Equalized_Odds", text="Model",
    title="模型性能与公平性权衡图",
    labels={"Error_Rate": "错误率 (1 - Accuracy)", "Equalized_Odds": "Equalized Odds 差异"}
)
fig_scatter.update_traces(textposition='top center')

###############################################
# 5. 构建交互式仪表盘（Dash 应用）
###############################################

# 创建 Dash 应用
app = dash.Dash(__name__)
app.title = "模型公平性分析仪表盘"

# 应用布局：使用 Tabs 展示各图表，同时提供一个下拉框查看不同模型的分组指标表格
app.layout = html.Div([
    html.H1("训练模型 + 公平性分析仪表盘", style={'textAlign': 'center'}),
    
    dcc.Tabs(id="tabs", value="tab-distribution", children=[
        dcc.Tab(label="分布图", value="tab-distribution"),
        dcc.Tab(label="柱状图", value="tab-bar"),
        dcc.Tab(label="ROC 曲线", value="tab-roc"),
        dcc.Tab(label="误差条图", value="tab-error"),
        dcc.Tab(label="性能-公平性散点图", value="tab-scatter"),
        dcc.Tab(label="分组指标表", value="tab-table"),
    ]),
    
    html.Div(id="tabs-content")
])

# 定义回调，根据选项显示对应图表
@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_content(tab):
    if tab == "tab-distribution":
        return html.Div([
            dcc.Graph(figure=fig_dist)
        ])
    elif tab == "tab-bar":
        return html.Div([
            dcc.Graph(figure=fig_bar)
        ])
    elif tab == "tab-roc":
        return html.Div([
            dcc.Graph(figure=fig_roc)
        ])
    elif tab == "tab-error":
        return html.Div([
            dcc.Graph(figure=fig_error)
        ])
    elif tab == "tab-scatter":
        return html.Div([
            dcc.Graph(figure=fig_scatter)
        ])
    elif tab == "tab-table":
        # 显示三个模型的分组指标（这里只显示基线模型示例）
        table_df = metric_frame_baseline.by_group.reset_index()
        return html.Div([
            html.H3("基线模型各群体指标："),
            dcc.Graph(
                figure=go.Figure(data=[go.Table(
                    header=dict(values=list(table_df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[table_df[col] for col in table_df.columns],
                               fill_color='lavender',
                               align='left'))
                ])
            ),
            html.Br(),
            html.H3("整体模型对比指标："),
            dcc.Graph(
                figure=go.Figure(data=[go.Table(
                    header=dict(values=list(results_df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[results_df[col] for col in results_df.columns],
                               fill_color='lavender',
                               align='left'))
                ])
            )
        ])
    else:
        return html.Div([html.H3("请选择一个标签页")])

###############################################
# 6. 运行 Dash 应用
###############################################
if __name__ == '__main__':
    app.run_server(debug=True)
