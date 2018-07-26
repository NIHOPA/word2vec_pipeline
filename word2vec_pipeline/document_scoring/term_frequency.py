# -*- coding: utf-8 -*-
'''
Builds the TF, TDF and discriminating_counter database for quick reference.
'''
import os
import pandas as pd
import random
import collections
from tqdm import tqdm
from bounter import bounter

class frequency_counter(object):
    function_name = "frequency_counter"

    def __init__(
            self,
            bounter_size_mb=1024,
            *args, **kwargs):
        # Global counter for term frequency
        self.TF = bounter(size_mb=bounter_size_mb)

    def __call__(self, row):
        raise NotImplementedError

    def save(self, output_data_directory, f_db, **kwargs):

        f_csv = os.path.join(output_data_directory, f_db)
        key_vals = [(x, self.TF[x]) for x in self.TF]

        df = pd.DataFrame(key_vals, columns=["word", "count"])
        df = df.sort_values("count", ascending=False)
        df.to_csv(f_csv, index=False)


class term_frequency(frequency_counter):

    function_name = "term_frequency"

    def __call__(self, row):
        '''
        Count frequency of terms in a single document and updates the
        class counters.

        Args:
            row (dict): Operates on the data in row['text']

        Returns:
            None
        '''

        text = row['text']

        tokens = unicode(text).split()
        self.TF.update(tokens)
        self.TF.update(["__pipeline_document_counter", ])


class term_document_frequency(frequency_counter):

    function_name = "term_document_frequency"

    def __call__(self, row):
        '''
        Count apperance of terms in a single document and updates the
        class counters.

        Args:
            row (dict): Operates on the data in row['text']

        Returns:
            None
        '''

        text = row['text']

        # For document frequency keep only the unique items
        tokens = set(unicode(text).split())

        self.TF.update(tokens)
        self.TF.update(["__pipeline_document_counter", ])
        

class discriminating_counter(list):
    '''
    Estimates, for each word w, the probability P(w ∈ D1 | w ∈ D2)
    '''
    function_name = "discriminating_counter"
    
    
    def __init__(
            self,
            sample_fraction,
            *args, **kwargs):

        self.sample_fraction = sample_fraction
        self.XOR = collections.Counter()
        self.OR = collections.Counter()

    def __call__(self, row):
        self.append( set(row['text'].split()) )

    def sample(self):
        x, y = random.sample(self, 2)
        self.XOR.update(x^y)
        self.OR.update(x|y)

    def save(self, output_data_directory, f_db, **kwargs):

        ITR = range(int(len(self)*self.sample_fraction))
        for n in tqdm(ITR):
            self.sample()

        df = pd.DataFrame(index=self.OR.keys(),)
        df.ix[self.XOR.keys(), "XOR"] = self.XOR.values()
        df.ix[self.OR.keys(),  "OR"] = self.OR.values()

        df["discriminating_factor"] = (df.XOR/df.OR)

        # Drop all words with a discriminating factor == 1 for speed
        df = df[~(df["discriminating_factor"] == 1)]

        del df["OR"]
        del df["XOR"]

        f_csv = os.path.join(output_data_directory, f_db)
        df = df.sort_values("discriminating_factor", ascending=True)
        df.index.name = 'word'

        df = df.dropna()
        df.to_csv(f_csv)    


