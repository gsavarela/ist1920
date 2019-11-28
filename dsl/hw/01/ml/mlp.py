""" This is an implementation of the backpropagation
    for feed forward network

    REFERENCE
    ---------
    Tom Mitchell, Machine Learning, March 1, 1997
"""
import numpy as np
from IPython.core.debugger import Pdb

debugger = Pdb()

def update_weights(W, V, X, T, lr=0.02):
    """Backpropagation

    """
    _, D = X.shape  # D input features
    M, _ = V.shape  # M hidden units
    N, K = T.shape  # N number of examples, K classes

    b = np.ones((N, 1), dtype=np.float32)
    X1 = np.concatenate((b, X), axis=1)
    # stochastic gradient decent
    indexes = np.random.choice(N, N, replace=False)
    for i in indexes:
        Xi = X1[i, :]
        Ti = T[i, :]

        H = forward(W, Xi, sigmoid)
        Z = forward(V, H, softmax)
        # hidden to outputs layer
        dV = Z * (1 - Z) * (Ti - Z)

        # input to hidden layer
        dW = H * (1 - H) * np.dot(V, dV)

        W += lr * dW
        V += lr * dV
    return W, V


def predict(W, V, X):
    """Performs forward propagation to a 2-layers
        feed-forward ann with sigmoid activation
        and softmax layer

    PARAMETERS:
    ----------
        * W numpy.array of floats
        it's the weights for the input layer, of size NxD
        * V numpy.array of floats
        it's the weithts for the hidden layer of size NxK

    RETURNS:
    --------
        * Z numpy.array of floats
        a propagated input
    """
    H = forward(W, X, sigmoid)
    Z = forward(V, H, softmax)
    Y = logits(Z)
    return Y


def logits(Z):
    Y = np.zeros(Z.shape, dtype=np.int16)
    Y[np.argmax(Z)] = 1
    return Y


def forward(W, X, fn):
    # TODO: extend to the case where X is a matrix
    # TODO: extend to put one dimension
    return fn(np.dot(W.T, X))


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

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def softmax(Z):
    expZ = np.exp(Z)
    return  expZ / np.sum(expZ)
