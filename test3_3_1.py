import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define neural network parameters
n_input = 784  # Number of input nodes (28*28)
n_hidden_1 = 256  # Number of nodes in the first hidden layer
n_classes = 10  # Number of output nodes (0-9)

# Define input and output
X = tf.compat.v1.placeholder("float", [None, n_input])
Y = tf.compat.v1.placeholder("float", [None, n_classes])

# Define weights and biases
weights = {
    'w1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}

# Define neural network model
def neural_net(x):
    # First hidden layer
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    # Output layer
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Build model
logits = neural_net(X)

# Define loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# Define optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)

# Initialize variables
init = tf.compat.v1.global_variables_initializer()

# Train model
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for epoch in range(10):
        avg_cost = 0.
        total_batch = int(len(x_train) / 100)
        for i in range(total_batch):
            batch_x = x_train[i*100:(i+1)*100].reshape(-1, 784)
            batch_y = y_train[i*100:(i+1)*100]
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch
        print(f"Epoch: {epoch + 1}, Cost: {avg_cost}")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test Accuracy:", accuracy.eval({X: x_test.reshape(-1, 784), Y: y_test}))