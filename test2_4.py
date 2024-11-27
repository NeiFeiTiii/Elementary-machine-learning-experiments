import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()

# (1) Use only the first two features for visualization
X = data.data[:, :2]
y = data.target  # Labels

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
regression_model = LogisticRegression(max_iter=10000)

# (2) Train the model
regression_results = regression_model.fit(X_train, y_train)

# (3) Predict the test set
y_predict = regression_results.predict(X_test)

# (4) Calculate parameters:
theta_0 = regression_model.intercept_[0]
theta_1 = regression_model.coef_[0]

# Output model coefficients
print(f"θ0 = {theta_0:.4f}")
print(f"θ1 = {theta_1}")

# Output decision boundary equation
print(f"Decision boundary equation: P(y=1|x) = 1 / (1 + exp(-({theta_0:.4f} + {theta_1[0]:.4f} * x1 + {theta_1[1]:.4f} * x2)))")