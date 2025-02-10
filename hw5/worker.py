def train_val_split(data, lmbd, train_index, test_index, kernel, beta, train, predict, loss):
    ker = kernel(beta=beta)
    train_X, test_X = data.X_train[train_index], data.X_train[test_index]
    train_y, test_y = data.y_train[train_index], data.y_train[test_index]
    alpha = train(ker, lmbd, train_X, train_y)
    preds = predict(ker, test_X, alpha, train_X)
    return loss(preds, test_y)