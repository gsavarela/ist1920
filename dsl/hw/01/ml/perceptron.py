""" This is an implementation of the multi-class perceptron

    REF:
    ---
    Perceptron algorithm, Rosemblat (1958)
"""
import numpy as np


def update_weights(X, Y, W=None):
    if W is None:
        Wshape = (Y.shape[1], X.shape[1])
        W = np.zeros(Wshape, dtype=np.double)

    mistakes = 0
    for i, Xi in enumerate(X):
        Yi = Y[i, :]
        Yp = predict(W, Xi)
        if not np.all(np.equal(Yp, Yi)):
            W += np.outer(Yi, Xi) - np.outer(Yp, Xi)
            mistakes += 1

    return W, mistakes


def predict(W, X):
    if len(X.shape) == 1:
        Yp = onehot_argmax(W, X)
    else:
        # Is there any better way?
        Yp = np.vstack([
            onehot_argmax(W, xi)
            for xi in X])
    return Yp


def evaluate(Yhat, Y):
    N, _ = Y.shape
    hits = np.all(np.equal(Yhat, Y), axis=1)
    return round(np.sum(hits) / N, 4)


def onehot_argmax(W, xi):
    K, D = W.shape
    yhat = np.argmax(np.dot(W, xi))
    Yp = np.identity(K)[:, yhat]
    return Yp

