import numpy as np

# Activation function: Step function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(x, w, Theta):
    return step_function(np.dot(x, w) - Theta)

# Logical AND operation
def logical_and(x1, x2):
    w = np.array([1, 1])
    Theta = 1.5
    return perceptron(np.array([x1, x2]), w, Theta)

# Logical OR operation
def logical_or(x1, x2):
    w = np.array([1, 1])
    Theta = 0.5
    return perceptron(np.array([x1, x2]), w, Theta)

# Logical NOT operation
def logical_not(x):
    w = np.array([-1])
    Theta = -0.5
    return perceptron(np.array([x]), w, Theta)

# Test the operations
print("AND Operation:")
print(f"0 AND 0 = {logical_and(0, 0)}")
print(f"0 AND 1 = {logical_and(0, 1)}")
print(f"1 AND 0 = {logical_and(1, 0)}")
print(f"1 AND 1 = {logical_and(1, 1)}")

print("\nOR Operation:")
print(f"0 OR 0 = {logical_or(0, 0)}")
print(f"0 OR 1 = {logical_or(0, 1)}")
print(f"1 OR 0 = {logical_or(1, 0)}")
print(f"1 OR 1 = {logical_or(1, 1)}")

print("\nNOT Operation:")
print(f"NOT 0 = {logical_not(0)}")
print(f"NOT 1 = {logical_not(1)}")