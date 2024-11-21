import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# Step 1: Load Data
def load_data():
    x1 = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]).reshape(-1, 1)
    y1 = np.array([-0.070676, -0.26191242, 0.17673216, 0.46497056, 0.87099123])

    x2 = np.linspace(0, 2 * np.pi, 21).reshape(-1, 1)
    y2 = np.array([-0.070676, 0.62069094, 0.9563237, -0.03169173, 2.83525242,
                   -0.26191242, -0.76075376, -1.60192248, -0.88072372, -1.8541696,
                   0.17673216, -1.93444425, 0.20282296, -1.41915353, -1.65908809,
                   0.46497056, -0.24124491, -0.47454724, 0.55812574, 1.9597344,
                   0.87099123])

    return (x1, y1), (x2, y2)


# Step 2: Construct Polynomial Model
class Polynomial_model:
    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        self.model = None

    def fit(self, x, y, model_type='linear'):
        x_poly = self.poly.fit_transform(x)
        if model_type == 'linear':
            self.model = LinearRegression().fit(x_poly, y)
        elif model_type == 'ridge':
            self.model = Ridge().fit(x_poly, y)
        elif model_type == 'lasso':
            self.model = Lasso().fit(x_poly, y)
        else:
            raise ValueError("Invalid model type")

    def score(self, x, y):
        x_poly = self.poly.transform(x)
        return self.model.score(x_poly, y)

    def predict(self, x):
        x_poly = self.poly.transform(x)
        return self.model.predict(x_poly)

    def getParam(self):
        return self.model.coef_, self.model.intercept_


# Step 3: Instantiate Polynomial Model
def fit_and_plot(ax, x, y, degree, model_type, color):
    model = Polynomial_model(degree)
    model.fit(x, y, model_type)
    y_pred = model.predict(x)

    # Step 5: Calculate score and mse
    score = model.score(x, y)
    mse = mean_squared_error(y, y_pred)

    # Step 6: Calculate polynomial coefficients
    coef, intercept = model.getParam()

    # Step 7: Plot fitting graph
    ax.scatter(x, y, color='blue', label='Data' if degree == 2 else "")
    ax.plot(x, y_pred, color=color, label=f'Degree {degree}')
    ax.set_title(f'{model_type.capitalize()} Regression')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    return coef, intercept, score, mse


# Load data
data1, data2 = load_data()

# Fit and plot for data1
fig, ax = plt.subplots()
colors = ['red', 'green', 'orange', 'purple']
#plt.figure()
fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data1 - Linear Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data1[0], data1[1], degree, 'linear', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()
#plt.figure()
fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data1 - Linear Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data1[0], data1[1], degree, 'lasso', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()
#plt.figure()
fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data1 - Linear Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data1[0], data1[1], degree, 'ridge', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()
# Fit and plot for data2
fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data2 - Linear Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data2[0], data2[1], degree, 'linear', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()

fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data2 - Lasso Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data2[0], data2[1], degree, 'lasso', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()

fig, ax = plt.subplots()
for degree, color in zip([2, 3, 4, 5], colors):
    print(f"Data2 - Ridge Regression (Degree {degree})")
    coef, intercept, score, mse = fit_and_plot(ax, data2[0], data2[1], degree, 'ridge', color)
    print(f"Coefficients: {coef}, Intercept: {intercept}, Score: {score}, MSE: {mse}\n")
plt.show()
plt.close()
