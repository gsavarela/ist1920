""" This is an implementation of the logistic regression

"""
import numpy as np


def update_weights(X, Y, lr=0.02, W=None):
    _, D = X.shape
    N, K = Y.shape
    if W is None:
        Wshape = (K, D + 1)
        W = np.zeros(Wshape, dtype=np.double)

    b = np.ones((N, 1), dtype=np.float32)
    X1 = np.concatenate((b, X), axis=1)
    # stochastic gradient decent
    indexes = np.random.choice(N, N, replace=False)
    for i in indexes:
        Xi = X1[i, :]
        Yi = Y[i, :]

        # P(y|x) = Q(y|x) / Zx
        Qyx = np.vstack([
            np.exp(
                np.dot(W[np.ravel(Yk), :], Xi)
            ) for Yk in np.identity(K).astype(bool)
        ])
        # Zx = sum Q(y|x) w.r.t ys
        Zx = np.sum(Qyx)
        Pyx = Qyx / Zx
        # This is the feature map for each output
        phiXY = np.repeat(Xi.reshape((1, D + 1)), K, axis=0)
        phiXiYi = np.outer(Yi, Xi)
        # Rescaled probabilities (allow for element-wise *)
        PYX = np.repeat(Pyx, D + 1, axis=1)
        W -= lr * (PYX * phiXY - phiXiYi)
    return W


def predict(W, X):
    if len(X.shape) == 1:
        Yp = onehot_argmax(W, X)
    else:
        preds = [onehot_argmax(W, xi)
                 for xi in X]
        Yp = np.vstack(preds)
    return Yp


def evaluate(Yhat, Y):
    N, _ = Y.shape
    hits = np.all(np.equal(Yhat, Y), axis=1)
    return round(np.sum(hits) / N, 4)


def onehot_argmax(W, xi):
    # xi is always a vector
    K, D = W.shape
    d = xi.shape[0]
    if len(xi.shape) == 1:
        xi = xi.reshape((d, 1))

    if d != D:
        bias = np.ones((1, 1), dtype=np.float32)
        xi = np.concatenate((bias, xi), axis=0)

    yhat = np.argmax(np.dot(W, xi))
    Yp = np.identity(K)[:, yhat]
    return Yp

