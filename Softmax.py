def softmax(x):
    import numpy as np
    exp_x = np.exp(x - np.max(x))  # 防止溢出
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)