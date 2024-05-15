import numpy
import numpy as np

from Cross_Entropy import calculate_cross_entropy
from RELU import relu_prime, relu
from Sigmoid import sigmoid
from Softmax import softmax
from matplotlib import pyplot as plt

################################
N = 5
OUT=10
epochs = 10000
N1=60
N2=120
acc_bst = 0.0
loss = []
accuracy = []
x_d = []

#################################

def draw(y2, y1, x):
    plt.figure(figsize=(13, 6))
    plt.plot(x, y1, label='loss', linestyle="--", color="orange")
    plt.plot(x, y2, label='accuracy', linestyle="dashdot", color="red")
    plt.xlabel("epoch")
    plt.ylabel("loss&acc")
    plt.title("CE and ACC wrt Epoch")
    plt.legend()
    plt.show()
    plt.pause(0.1)
# 神经网络模型
def multi_class(W1, W2, W3, X, D, N):
    from Softmax import softmax
    from Sigmoid import sigmoid
    import numpy as np
    alpha = 0.01

    for k in range(N):
        x = np.reshape(X[:, :, k], (25, 1))
        d = D[k, :].reshape(-1, 1)
        v1 = np.dot(W1, x)
        y1 = relu(v1)
        v2 = np.dot(W2, y1)
        y2 = relu(v2)
        v3 = np.dot(W3, y2)
        y = softmax(v3)
        # print(y)
        e = d - y
        delta = e

        # 反向传播
        e2 = np.dot(W3.T, delta)
        delta2 = relu_prime(v2) * e2

        e1 = np.dot(W2.T, delta2)
        delta1 = relu_prime(v1) * e1

        dW1 = alpha * np.dot(delta1, x.T)
        W1 = W1 + dW1

        dW2 = alpha * np.dot(delta2, y1.T)
        W2 = W2 + dW2

        dW3 = alpha * np.dot(delta, y2.T)
        W3 = W3 + dW3

    return W1, W2, W3


######################################################
X = np.zeros((5, 5, 10))

X[:, :, 0] = [[0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0]]

X[:, :, 1] = [[1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1]]

X[:, :, 2] = [[1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0]]

X[:, :, 3] = [[0, 0, 0, 1, 0],
              [0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 1, 0]]

X[:, :, 4] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 0]]

X[:, :, 5] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

X[:, :, 6] = [[1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1]]

X[:, :, 7] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

X[:, :, 8] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

X[:, :, 9] = [[1, 0, 1, 1, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 1, 1]]
D = np.eye(10)

######################################################

######################################################
Y = np.zeros((5, 5, 10))

Y[:, :, 0] = [[0, 0, 1, 1, 0],
              [0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]]

Y[:, :, 1] = [[1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

Y[:, :, 2] = [[1, 1, 1, 1, 0],
              [0, 0, 0, 0, 1],
              [0, 1, 1, 1, 0],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 0]]

Y[:, :, 3] = [[0, 1, 1, 1, 0],
              [0, 1, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [0, 1, 1, 1, 0]]

Y[:, :, 4] = [[0, 1, 1, 1, 1],
              [0, 1, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0],
              [1, 1, 1, 1, 0]]

Y[:, :, 5] = [[1, 1, 1, 1, 1],
              [1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

Y[:, :, 6] = [[1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1]]

Y[:, :, 7] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1]]

Y[:, :, 8] = [[1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 0, 1],
              [1, 1, 1, 1, 1]]

Y[:, :, 9] = [[1, 0, 1, 1, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 1, 1],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 1, 1]]

D_Y = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]
######################################################

W1 = 2 * np.random.rand(N1, 25) - 1
W2 = 2 * np.random.rand(N2, N1) - 1
W3 = 2 * np.random.rand(OUT, N2) - 1

W1_bst = np.zeros([N1, 25], dtype=np.float32)
W2_bst = np.zeros([N2, N1], dtype=np.float32)
W3_bst = np.zeros([OUT, N2], dtype=np.float32)

######################################################
# Train
for epoch in range(epochs):
    W1, W2, W3 = multi_class(W1, W2, W3, X, D, N)
    loss.append(calculate_cross_entropy(W1, W2, W3, X, D, N))
    x_d.append(epoch)

    # Test
    acc = 0.0
    correct_cnt = 0
    for k in range(N):
        x = np.reshape(Y[:, :, k], (25, 1))
        label = np.argmax(D_Y[k]) + 1
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2, y1)
        y2 = sigmoid(v2)
        v3 = np.dot(W3, y2)
        y3 = sigmoid(v3)
        y = softmax(y3)
        Predicted = numpy.argmax(y) + 1
        if Predicted == label:
            correct_cnt = correct_cnt + 1
        # print(f"Predicted:{Predicted} Label:{label}")

    acc = correct_cnt / N
    if acc > acc_bst:
        acc_bst = acc
        W1_bst = W1
        W2_bst = W2
        W3_bst = W3
    accuracy.append(acc)

draw(accuracy, loss, x_d)

# Inference
print(f"Best Accuracy:{acc_bst}")
for k in range(N):
    x = np.reshape(Y[:, :, k], (25, 1))
    label = np.argmax(D_Y[k]) + 1
    v1 = np.dot(W1_bst, x)
    y1 = sigmoid(v1)
    v2 = np.dot(W2_bst, y1)
    y2 = sigmoid(v2)
    v3 = np.dot(W3_bst, y2)
    y3 = sigmoid(v3)
    y = softmax(y3)
    Predicted = numpy.argmax(y) + 1
    print(f"Predicted Vector :\n {y}")
    print(f"Predicted:{Predicted} Label:{label}")
