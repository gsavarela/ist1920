"""This script helps provides utilities for performing a 
    Transliteration Task

    URL:
    ---
        https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-train.txt
        https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-eval.txt
        https://https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-test.txt
"""

import pandas as pd

import torch
from torch.utils.data import Dataset


def tokenize(df):
    pass


class AR2EN(Dataset):
    """Dataset Arabic-to-English provided by Google

        REFERENCE:
        ---------
        https://pytorch.org/docs/stable/torchvision/datasets.html#id17
        """
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        """"Instanciates an Arabic-to-English Dataset

        PARAMETERS:
        -----------
        * root (string) – Root directory of dataset where MNIST/processed/training.pt and MNIST/processed/test.pt exist.

        * train (bool, optional) – If True, creates dataset from training.pt, otherwise from test.pt.

        * download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

        * transform (callable, optional) – A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop

        * target_transform (callable, optional) – A function/transform that takes in the target and transforms it.
        """

        if train:
            # load train & eval
            df_train = pd.read_csv(f'{root}ar2en-train.txt',
                                   sep='\t', encoding='utf-8', header=None)
            df_eval = pd.read_csv(f'{root}ar2en-eval.txt',
                                  sep='\t', encoding='utf-8', header=None)

            df = pd.concat((df_train, df_eval), axis=0)

            self.data = torch.tensor(df[0].values)
            self.targets = torch.tensor(df[1].values)
        else:
            # load test
            pass

        if download:
            raise NotImplementedError


if __name__ == '__main__':
    data = AR2EN('')

    X = data.data
    Y = data.targets
    import pdb
    pdb.set_trace()
