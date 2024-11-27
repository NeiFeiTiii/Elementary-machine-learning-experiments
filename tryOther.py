import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 加载iris数据集
data = load_iris()
X = data.data  # 特征
y = data.target  # 标签

# 将数据划分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Logistic回归模型（多分类）
model = LogisticRegression(max_iter=10000, multi_class='ovr')

# 训练模型
model.fit(X_train, y_train)

# 获取模型系数
theta_01, theta_11, theta_21 = model.intercept_[0], model.coef_[0][0], model.coef_[0][1]
theta_02, theta_12, theta_22 = model.intercept_[1], model.coef_[1][0], model.coef_[1][1]
theta_03, theta_13, theta_23 = model.intercept_[2], model.coef_[2][0], model.coef_[2][1]

# 输出模型系数
print(f"θ01 = {theta_01:.4f}, θ11 = {theta_11:.4f}, θ21 = {theta_21:.4f}")
print(f"θ02 = {theta_02:.4f}, θ12 = {theta_12:.4f}, θ22 = {theta_22:.4f}")
print(f"θ03 = {theta_03:.4f}, θ13 = {theta_13:.4f}, θ23 = {theta_23:.4f}")

# 分类平面方程
print(f"分类平面方程1： P(y=0|x) = 1 / (1 + exp(-(θ01 + θ11 * x1 + θ21 * x2)))")
print(f"分类平面方程2： P(y=1|x) = 1 / (1 + exp(-(θ02 + θ12 * x1 + θ22 * x2)))")
print(f"分类平面方程3： P(y=2|x) = 1 / (1 + exp(-(θ03 + θ13 * x1 + θ23 * x2)))")

# 选择前两个特征用于可视化
X_vis = X[:, :2]  # 只选择前两个特征
y_vis = y  # 标签

# 创建一个网格来绘制分类结果
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 使用训练数据的前两个特征进行预测
# 将网格输入扩展为四个特征的维度，保持与训练模型一致
grid_points = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())]  # 补充后两个特征
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)

# 绘制分类平面
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# 绘制训练数据点
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k', marker='o', cmap=plt.cm.coolwarm, label='Train Data')

# 设置标签和标题
plt.xlabel('Feature 1: Sepal length')
plt.ylabel('Feature 2: Sepal width')
plt.title('Logistic Regression - Iris Dataset')

# 添加图例
plt.legend()

# 显示图像
plt.show()

