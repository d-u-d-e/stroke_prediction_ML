def error_rate(y_true, y_pred):
    size = y_true.size
    miss_samples = 0
    for i in range(size):
        if y_pred[i] != y_true[i]:
            miss_samples = miss_samples + 1
    return miss_samples / size
