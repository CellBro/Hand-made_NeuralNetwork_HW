


def calculate_mse(W1, W2, W3, X, D,N):
    from Softmax import softmax
    from Sigmoid import sigmoid
    import numpy as np
    mse = []

    for k in range(N):
        x = np.reshape(X[:, :, k], (25, 1))
        d = D[k, :].reshape(-1, 1)
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2, y1)
        y2 = sigmoid(v2)
        v3 = np.dot(W3, y2)
        y3 = softmax(v3)
        e = d - y3
        mse.append(np.mean(e**2))  # 计算每个样本的均方误差

    return np.mean(mse)  # 返回平均均方误差