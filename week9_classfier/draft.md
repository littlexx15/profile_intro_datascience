如何用这个学生成绩数据集去完成所要求的三个任务——即 (1) 训练 AdaBoost 并比较它与随机森林模型，(2) 调整 AdaBoost 超参数，(3) 进行不同的特征工程实验——从而提升模型准确率并理解其中原理。

1. 明确目标与数据
目标：

预测学生的“Grade”列（A/B/C/D/F），这是一个多分类问题。
也可自定义二分类，比如：A/B=“通过”，C/D/F=“不通过”。
数据集初探：

该数据包含了学生的个人信息（Name、Email 等）、学习表现（Midterm_Score、Final_Score、Assignments_Avg 等）以及家庭背景（Parent_Education_Level、Family_Income_Level 等）。
你需要先查看哪些特征有助于预测成绩，哪些列可以直接删掉（如 Student_ID、Email 等对成绩预测意义不大）。
