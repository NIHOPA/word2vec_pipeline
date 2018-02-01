'''
Builds the Term Frequency database for quick reference.
'''
import collections
import os

import pandas as pd
from utils.mapreduce import simple_mapreduce


class frequency_counter(simple_mapreduce):
    table_name = None

    def __init__(self, *args, **kwargs):
        '''
        Initialize counter

        Args:
            args: DOCUMENTATION_UNKNOWN
            kwargs: DOCUMENTATION_UNKNOWN
        '''

        # Global counter for term frequency
        self.TF = collections.Counter()
        super(frequency_counter, self).__init__(*args, **kwargs)

    def reduce(self, C):
        '''
        Update counter

        Args:
            C: term [possible rename to something more informative]
        '''
        self.TF.update(C)

    def report(self):
        '''
        Return counter
        '''
        return self.TF

    def save(self, config):
        '''
        Save counter

        Args:
            config: config file
        '''
        f_csv = os.path.join(
            config["output_data_directory"],
            config[self.table_name]["f_db"])

        df = pd.DataFrame(self.TF.most_common(),
                          columns=["word", "count"])

        df.set_index('word').to_csv(f_csv)


class term_frequency(frequency_counter):
    '''
    Count frequency of terms
    DOCUMENTATION_UNKNOWN - Is this for all documents, or a single document?
    '''

    table_name = "term_frequency"

    def __call__(self, row):
        '''
        Count frequency of terms in a single document

        Args:
            row: row of a pandas dataframe representing a single document

        Returns:
            C: a counter of term frequency in the graph
        '''
        text = row['text']

        tokens = unicode(text).split()
        C = collections.Counter(tokens)

        # Add a token to keep track of total documents
        C["__pipeline_document_counter"] += 1

        return C


class term_document_frequency(frequency_counter):
    '''
    Count frequency of terms
    DOCUMENTATION_UNKNOWN - Is this for all documents, or a single document?
    '''

    table_name = "term_document_frequency"

    def __call__(self, row):
        '''
        Count frequency of terms in a single document

        Args:
            row: row of a pandas dataframe representing a single document

        Returns:
            C: a counter of term frequency in the graph
        '''
        text = row['text']

        # For document frequency keep only the unique items
        tokens = set(unicode(text).split())
        C = collections.Counter(tokens)

        # Add an empty string token to keep track of total documents
        C["__pipeline_document_counter"] += 1

        return C
