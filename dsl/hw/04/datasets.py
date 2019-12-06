"""This script helps provides utilities for performing a 
    Transliteration Task


    Extends torchtext.data

    'The data module provides the following:

    Ability to define a preprocessing pipeline
    Batching, padding, and numericalizing
    (including building a vocabulary object)
    Wrapper for dataset splits (train, validation, test)
    Loader a custom NLP dataset'


    REFERENCE:
    ---------
        https://torchtext.readthedocs.io/en/latest/data.html

    URL:
    ---
        https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-train.txt
        https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-eval.txt
        https://https://raw.githubusercontent.com/googlei18n/transliteration/master/ar2en-test.txt
"""

import pandas as pd

import torch
from torch.utils.data import Dataset
from torchtext.data import TabularDataset, Field

def tokenize(df):
    pass


class AR2EN(TabularDataset):
    """Dataset Arabic-to-English provided by Google

        REFERENCE:
        ---------

        https://torchtext.readthedocs.io/en/latest/data.html#tabulardataset
        """
    def __init__(self):
        """"Instanciates an Arabic-to-English Dataset


            Create a TabularDataset given a path, file format, and field list.

        PARAMETERS:
        -----------
        * path: (str)
            Path to the data file.

        * format: (str)
            The format of the data file. One of 'CSV', 'TSV',
            or 'JSON' (case-insensitive).

        * fields: (list(tuple(str, Field))
            tuple(str, Field)]: If using a list, the format must
            be CSV or TSV, and the values of the list should be
            tuples of (name, field). The fields should be in the
            same order as the columns in the CSV or TSV file,
            while tuples of (name, None) represent columns that
            will be ignored. If using a dict, the keys should be
            a subset of the JSON keys or CSV/TSV columns, and the
            values should be tuples of (name, field). Keys not
            present in the input dictionary are ignored. This
            allows the user to rename columns from their
            JSON/CSV/TSV key names and also enables selecting
            a subset of columns to load.

        * skip_header: (bool)
            Whether to skip the first line of the input file.
        * csv_reader_params:(dict)
            Parameters to pass to the csv reader. Only relevant
            when format is csv or tsv. See 
            https://docs.python.org/3/library/csv.html#csv.reader
            for more details.

        """
        super(TabularDataset, self).__init__('ar2en-train.txt',
                                      format='CSV',
                                      fields=[('ARABIC', Field()),
                                              ('ENGLISH', Field())])

if __name__ == '__main__':
    """
     class torchtext.data.Field(sequential=True, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.int64, preprocessing=None, postprocessing=None, lower=False, tokenize=None, tokenizer_language='en', include_lengths=False, batch_first=False, pad_token='<pad>', unk_token='<unk>', pad_first=False, truncate_first=False, stop_words=None, is_target=False)

    """
    TEXT = Field()
    TARGET = Field()
    train, valid, test = TabularDataset.splits(
        path='',
        train='ar2en-train.txt',
        validation='ar2en-eval.txt',
        test='ar2en-test.txt',
        format='csv',
        fields=[('ARABIC', TEXT),
                ('ENGLISH', TARGET)]
    )

    import pdb
    pdb.set_trace()
