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

# Logical NAND operation (NOT AND)
def logical_nand(x1, x2):
    w = np.array([-1, -1])
    Theta = -1.5
    return perceptron(np.array([x1, x2]), w, Theta)

# XOR operation using two-layer perceptron network
def logical_xor(x1, x2):
    nand_result = logical_nand(x1, x2)
    or_result = logical_or(x1, x2)
    return logical_and(nand_result, or_result)

# Test the XOR operation
print("XOR Operation:")
print(f"0 XOR 0 = {logical_xor(0, 0)}")
print(f"0 XOR 1 = {logical_xor(0, 1)}")
print(f"1 XOR 0 = {logical_xor(1, 0)}")
print(f"1 XOR 1 = {logical_xor(1, 1)}")


# Draw......
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes for the first layer (inputs and hidden layer)
G.add_node("x1", pos=(0, 1))
G.add_node("x2", pos=(0, -1))
G.add_node("NAND", pos=(1, 1))
G.add_node("OR", pos=(1, -1))

# Add nodes for the second layer (output layer)
G.add_node("AND", pos=(2, 0))
G.add_node("output", pos=(3, 0))

# Add edges for the first layer
G.add_edge("x1", "NAND", label="w1")
G.add_edge("x2", "NAND", label="w2")
G.add_edge("x1", "OR", label="w1")
G.add_edge("x2", "OR", label="w2")

# Add edges for the second layer
G.add_edge("NAND", "AND", label="w1")
G.add_edge("OR", "AND", label="w2")
G.add_edge("AND", "output")

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(figsize=(10, 5))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("Two-Layer Perceptron Structure for XOR Operation")
plt.show()
