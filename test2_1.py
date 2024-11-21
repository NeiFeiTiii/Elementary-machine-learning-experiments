import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 示例数据
x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([0, 0.69, 1.10, 1.39, 1.61, 1.79, 1.95, 2.08, 2.20, 2.30])

# 添加常数项以包含截距
x = sm.add_constant(x)

# 拟合模型 GLM 广义线性模型
model = sm.GLM(y, x, family=sm.families.Gaussian())
results = model.fit()

# 打印摘要
print(results.summary())

# 预测值
y_pred = results.predict(x)

# 绘制结果
plt.scatter(x[:, 1], y, color='blue', label='data')
plt.plot(x[:, 1], y_pred, color='red', label='Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 计算 t
a = results.params[1]  # 获取系数 a
x_values = np.arange(0.5, 5.5, 0.5)
for x_value in x_values:
    y_value = np.log(a * x_value)
    t = np.exp(y_value)
    print(f"求得 a = {a}")
    print(f"当 x = {x_value} 时，t = {t}")