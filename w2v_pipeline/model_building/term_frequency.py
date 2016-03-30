'''
Builds the TF database for quick reference.
'''
import itertools
import collections

import pandas as pd
import sqlalchemy

from mapreduce import simple_mapreduce

class term_frequency(simple_mapreduce):

    def __init__(self,*args,**kwargs):
        # Global counter for term frequency
        self.TF = collections.Counter()
        
        super(term_frequency, self).__init__(*args,**kwargs)

    def __call__(self, item):

        text = item[0]   
        tokens = unicode(text).split()
        C = collections.Counter(tokens)

        # Add an empty string token to keep track of total documents
        C[""] += 1

        return [C,] + item[1:]

    def reduce(self, C):
        self.TF.update(C)

    def report(self):
        return self.TF

    def save(self, config):
        
        df = pd.DataFrame(self.TF.most_common(),
                          columns=["word","count"])

        f_sql = config["term_frequency"]["f_db"]
        engine = sqlalchemy.create_engine('sqlite:///'+f_sql)

        df.to_sql("term_frequency", engine,
                  if_exists='replace')
