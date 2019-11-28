""" This is an implementation of logistic regression

"""
import matplotlib.pyplot as plt
import numpy as np
from ocr.process import get_lexicon, get_partition
from ml.mlp import (evaluate, update_weights,
                                    predict)


if __name__ == '__main__':
    lex = get_lexicon()
    Xtrain, Ttrain = get_partition("train", lex=lex)

    Xvalid, Tvalid = get_partition("valid", lex=lex)

    # Peceptron algorithm
    max_epochs = 20  # number of epochs
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ttrain', color='c')
    ax2.set_ylabel('Eval', color='b')
    train_acc = []
    valid_acc = []

    # instanciate the weights
    num_hidden = 256
    num_classes = Ttrain.shape[1]
    num_features = Xtrain.shape[1]
    W = np.random.randn(num_features + 1, num_hidden)
    V = np.random.randn(num_hidden, num_classes)

    for e in range(max_epochs):

        W, V = update_weights(W, V, Xtrain, Ttrain)

        Ytrain = np.vstack([predict(W, V, Xi) for Xi in Xtrain])
        acc = round(evaluate(Ytrain, Ttrain), 4)
        train_acc.append(acc)

        Yvalid = np.vstack([predict(W, V, Xi) for Xi in Xvalid])
        acc = round(evaluate(Yvalid, Tvalid), 4)
        valid_acc.append(acc)
        print(f"Epochs: {e}\t train(%): {train_acc[-1]} valid(%): {valid_acc[-1]}")
        # Print to the user the training results
        ax1.plot(train_acc, 'c-')
        ax2.plot(valid_acc, 'b-')

        plt.title("Multi-class multi-layer perceptron OCR")
        plt.draw()
        plt.pause(0.01)
 
    Xtest, Ytest = get_partition("test", lex=lex)
    print(f"Test accuracy {evaluate(predict(W, Xtest), Ytest)}")
