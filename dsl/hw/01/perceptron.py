""" This is an implementation of the multi-class perceptron

    REF:
    ---
    Perceptron algorithm, Rosemblat (1958)
"""
import matplotlib.pyplot as plt
import numpy as np
from ocr.process import get_lexicon, get_partition
from ml.perceptron import evaluate, update_weights, predict


if __name__ == '__main__':
    lex = get_lexicon()
    Xtrain, Ytrain = get_partition("train", lex=lex)

    Xvalid, Yvalid = get_partition("valid", lex=lex)

    # Peceptron algorithm
    max_epochs = 20  # number of epochs
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train', color='c')
    ax2.set_ylabel('Eval', color='b')
    train_acc = []
    valid_acc = []
    W = None
    for e in range(max_epochs):

        W, mistakes = update_weights(Xtrain, Ytrain, W=W)

        acc = round(evaluate(predict(W, Xtrain), Ytrain), 4)
        train_acc.append(acc)

        acc = evaluate(predict(W, Xvalid), Yvalid)
        valid_acc.append(acc)
        print(f"Epochs: {e}\t train(%): {train_acc[-1]} valid(%): {valid_acc[-1]}")
        # Print to the user the training results
        ax1.plot(train_acc, 'c-')
        ax2.plot(valid_acc, 'b-')

        plt.title("Multi-class perceptron OCR")
        plt.draw()
        plt.pause(0.01)
 
    Xtest, Ytest = get_partition("test", lex=lex)
    print(f"Test accuracy {evaluate(predict(W, Xtest), Ytest)}")
