# MSE = (1/n) * sum(y - y_hat)^2

def mse(y_true, y_pred):
    n = len(y_true)
    s = 0
    for i in range(n):
        diff = y_true[i] - y_pred[i]
        s += diff * diff  # (y - y_hat)^2
    return s / n
