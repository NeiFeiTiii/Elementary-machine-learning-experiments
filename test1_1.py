import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 数据1
x1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]).reshape(-1, 1)
y1 = np.array([1.01, 1.19, 1.42, 1.57, 1.83, 2.58, 3.38, 4.22, 5.01, 5.79])

# 拟合数据1
lr1 = LinearRegression()
lr1.fit(x1, y1)
a1 = lr1.coef_[0]
b1 = lr1.intercept_
y1_predict = lr1.predict(x1)

# 预测数据1
y1_pred_3 = lr1.predict(np.array([[3]]))[0]
y1_pred_6_5 = lr1.predict(np.array([[6.5]]))[0]

# 绘制数据1拟合图
plt.scatter(x1, y1, color='blue')
plt.plot(x1, y1_predict, color='red')
plt.title('Data 1 Linear Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 数据2
x2 = np.linspace(0, 10, 20).reshape(-1, 1)
y2 = 2 * x2.flatten() + 1 + np.random.normal(0, 1, x2.shape[0])  # 生成带噪声的y

# 拟合数据2
lr2 = LinearRegression()
lr2.fit(x2, y2)
a2 = lr2.coef_[0]
b2 = lr2.intercept_
y2_predict = lr2.predict(x2)

# 预测数据2
y2_pred_3 = lr2.predict(np.array([[3]]))[0]
y2_pred_9_5 = lr2.predict(np.array([[9.5]]))[0]

# 绘制数据2拟合图
plt.scatter(x2, y2, color='blue')
plt.plot(x2, y2_predict, color='red')
plt.title('Data 2 Linear Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 输出结果
print(f"数据1: a = {a1}, b = {b1}")
print(f"数据1: x = 3, y = {y1_pred_3}")
print(f"数据1: x = 6.5, y = {y1_pred_6_5}")

print(f"数据2: a = {a2}, b = {b2}")
print(f"数据2: x = 3, y = {y2_pred_3}")
print(f"数据2: x = 9.5, y = {y2_pred_9_5}")
