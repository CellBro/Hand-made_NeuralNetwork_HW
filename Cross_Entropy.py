import math


def calculate_cross_entropy(W1, W2, W3, X, D, N):
    from Sigmoid import sigmoid
    from Softmax import softmax
    import numpy as np
    cross_entropy = []

    for k in range(N):
        x = np.reshape(X[:, :, k], (25, 1))
        d = D[k, :].reshape(-1, 1)
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2, y1)
        y2 = sigmoid(v2)
        v3 = np.dot(W3, y2)
        y3 = softmax(v3)

        e = -1*np.log(y3[np.argmax(d)])
        # 计算交叉熵
        cross_entropy.append(e)

    return np.mean(cross_entropy)  # 返回平均交叉熵
