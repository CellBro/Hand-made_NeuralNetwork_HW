def sigmoid(x):
    import numpy as np
    # Clip input values to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
