""" This a helper script which handles the letter dataset

    OCR:
    ----

This dataset contains handwritten words dataset collected by Rob Kassel at MIT Spoken Language Systems Group. I selected a "clean" subset of the words and rasterized and normalized the images of each letter. Since the first letter of each word was capitalized and the rest were lowercase, I removed the first letter and only used the lowecase letters. The tab delimited data file (letter.data.gz) contains a line for each letter, with its label, pixel values, and several additional fields listed in letter.names file.

    FIELDS:
    -------

    * id: each letter is assigned a unique integer id
    * letter: a-z
    * next_id: id for next letter in the word, -1 if last letter
    * word_id: each word is assigned a unique integer id (not used)
    * position: position of letter in the word (not used)
    * fold: 0-9 -- cross-validation fold
    * p_i_j: 0/1 -- value of pixel in row i, column j 

    REF:
    ---
    http://ai.stanford.edu/~btaskar/ocr/
"""

import pandas as pd
import numpy as np

def get_lexicon():

    df = pd.read_csv("ocr/letter.data", sep="\t", header=None, index_col=0)
    df.head()
    #  extract letters
    letter_ids = \
        {letter_: id_
         for id_, letter_ in enumerate(set(df.iloc[:, 0].values))}

    return letter_ids

def get_partition(name, lex=None):
    '''Reads the letter.data returning a segment

        PARAMETERS:
        -----------
        name: string, obligatory
            partion name must be 'train', 'valid' or 'test'
        lex: dictionary, optional
            the letters to class ( integer ) mapping

        RETURNS:
        --------
        Returns a partition from the OCR dataset
        X: ndarray [num_observations, 128]
            A binary feature representation for letters having
            size 16x8 pixels
        Y: ndarray [num_observations, 26]
            A one-hot enconding for the 26 classes on OCR

        USAGE:
        ------
        Xtest, Ytest = get_partition("test")
    '''
    #  Input sanitization
    if name not in ('train', 'test', 'valid'):
        raise ValueError("name must be in a proper dataset partition")
    if lex is None:
        lex = get_lexicon()

    df = pd.read_csv("ocr/letter.data", sep="\t", header=None, index_col=0)
    #  Partion the dataset using column #4
    #  presenting the folds
    if name in ('train',):
        df = df[df.iloc[:, 4] < 8]
    elif name in ('valid',):
        df = df[df.iloc[:, 4] == 8]
    elif name in ('test',):
        df = df[df.iloc[:, 4] == 9]

    #  replaces letters by ids and return an array
    idy = df.iloc[:,0].map(lex).values
    idx = np.arange(idy.shape[0])
    Y = np.zeros((df.shape[0], len(lex)), dtype=int)
    # One-hot encoding the representations
    Y[idx, idy] = 1
    X = np.asarray(df.iloc[:, 5:-1].values)
    return X, Y


if __name__ == '__main__':
    lex = get_lexicon()
    print(f"Number of classes is {len(lex)}")

    Xtrain, Ytrain = get_partition("train", lex=lex)
    print(f"Is there any nan? {np.any(np.isnan(Xtrain))}")
    print(f"Training data shape is {Xtrain.shape}")
    assert Xtrain.shape[0] == 41679


    Xvalid, Yvalid = get_partition("valid", lex=lex)
    print(f"Validation data shape is {Xvalid.shape}")
    assert Xvalid.shape[0] == 5331

    Xtest, Ytest = get_partition("test", lex=lex)
    print(f"Test shape is {Xtest.shape}")
    assert Xtest.shape[0] == 5142




