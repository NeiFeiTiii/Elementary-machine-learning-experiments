import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Example data
x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([0, 0.69, 1.10, 1.39, 1.61, 1.79, 1.95, 2.08, 2.20, 2.30])

# Add noise to the data
noise = np.random.normal(0, 0.1, y.shape)
y_noisy = y + noise

# Apply logarithmic transformation to the features
x_log = np.log(x)

# Add constant term to include intercept
x_log = sm.add_constant(x_log)

# Fit model with log-transformed features
model = sm.GLM(y_noisy, x_log, family=sm.families.Gaussian(sm.families.links.identity()))
results = model.fit()

# Print summary
print(results.summary())

# Predicted values
y_pred = results.predict(x_log)

# Plot results
plt.scatter(x, y_noisy, color='blue', label='data with noise')
plt.plot(x, y_pred, color='red', label='Fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Calculate t
a = results.params[1]  # Get coefficient a
print(f"Calculated a = {a}")
x_values = np.arange(0.5, 5.5, 0.5)
for x_value in x_values:
    y_value = np.log(a * x_value)
    t = np.exp(y_value)

    print(f"When x = {x_value}, t = {t}")