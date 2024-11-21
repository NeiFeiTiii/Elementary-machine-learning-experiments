import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLassoCV
from sklearn.metrics import mean_squared_error


# Step 1: Load Data
def load_data(file_path):
    data = pd.read_excel('./data.xlsx')
    data = data.iloc[:, 1:]  # Remove the first column
    x = data.iloc[:, :-4].values  # Independent variables
    y = data.iloc[:, -4:].values  # Dependent variables
    return train_test_split(x, y, test_size=0.2, random_state=42)


# Step 2: Instantiate Regression Models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=0.5)
lasso_reg = MultiTaskLassoCV(cv=10)


# Step 3: Train and Evaluate Models
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    return score, mse, y_pred


# Step 4: Plot Fitting Graph
def plot_fitting(x_test, y_test, y_pred, title):
    plt.figure(figsize=(10, 6))
    for i in range(y_test.shape[1]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_test[:, 0], y_test[:, i], color='blue', label='Actual')
        plt.scatter(x_test[:, 0], y_pred[:, i], color='red', label='Predicted')
        plt.title(f'{title} - Component {i + 1}')
        plt.xlabel('Blow Oxygen Time')
        plt.ylabel(f'Component {i + 1}')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Main Function
def main(file_path):
    x_train, x_test, y_train, y_test = load_data(file_path)

    for model, name in zip([linear_reg, ridge_reg, lasso_reg],
                           ['Linear Regression', 'Ridge Regression', 'LASSO Regression']):
        score, mse, y_pred = train_and_evaluate(model, x_train, y_train, x_test, y_test)
        print(f'{name} - Score: {score}, MSE: {mse}')
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        result = model.predict(x_test)
        plt.figure()
        plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
        plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
        plt.title('score: %f' % score)
        plt.legend()
        plt.show()
        plot_fitting(x_test, y_test, y_pred, name)


# Run the main function with the path to your Excel file
main('./data.xlsx')
