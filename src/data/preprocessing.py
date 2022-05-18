def scale(y_train, y_test, scaling):
    # y: (signal_dim, T)
    if scaling == 'axiswise':
        mean = y_train.mean(dim=1, keepdims=True)
        std = y_train.std(dim=1, keepdims=True)
    else:
        mean = y_train.mean()
        std = y_train.std()

    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std

    return y_train, y_test


def train_test_split(t, y_clean, y, normalize_t, scaling, train_frac):
    # t: (T,)
    # y_clean: (signal_dim, T)
    # y: (signal_dim, T)
    N = ((t / t.max()) <= train_frac).sum()

    if normalize_t:
        t = t / t.max()

    t_train, t_test = t[:N], t[N:]
    y_clean_train, y_clean_test = y_clean[:, :N], y_clean[:, N:]
    y_train, y_test = y[:, :N], y[:, N:]

    y_clean_train, y_clean_test = scale(y_clean_train, y_clean_test, scaling)
    y_train, y_test = scale(y_train, y_test, scaling)

    return t_train, y_clean_train, y_train, t_test, y_clean_test, y_test
