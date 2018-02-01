'''
Builds the TF database for quick reference.
'''
import os
import pandas as pd
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
        text = row['text']

        tokens = unicode(text).split()
        self.TF.update(tokens)
        self.TF.update(["__pipeline_document_counter", ])


class term_document_frequency(frequency_counter):

    function_name = "term_document_frequency"

    def __call__(self, row):
        text = row['text']

        # For document frequency keep only the unique items
        tokens = set(unicode(text).split())

        self.TF.update(tokens)
        self.TF.update(["__pipeline_document_counter", ])
