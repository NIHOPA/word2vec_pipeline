'''
Builds the TF database for quick reference.
'''
import itertools
import collections
import os

import pandas as pd
import sqlalchemy

from utils.mapreduce import simple_mapreduce

class frequency_counter(simple_mapreduce):
    table_name = None

    def __init__(self,*args,**kwargs):
        
        # Global counter for term frequency
        self.TF = collections.Counter()       
        super(frequency_counter, self).__init__(*args,**kwargs)


    def reduce(self, C):
        self.TF.update(C)

    def report(self):
        return self.TF

    
    def save(self, config):

        df = pd.DataFrame(self.TF.most_common(),
                          columns=["word","count"])

        out_dir = config["output_data_directory"]
        f_sql = os.path.join(out_dir, config["term_frequency"]["f_db"])
        
        engine = sqlalchemy.create_engine('sqlite:///'+f_sql)

        df.to_sql(self.table_name,
                  engine,
                  if_exists='replace')
    

class term_frequency(frequency_counter):

    table_name = "term_frequency"

    def __call__(self, item):

        text = item[0]   
        tokens = unicode(text).split()
        C = collections.Counter(tokens)

        # Add an empty string token to keep track of total documents
        C[""] += 1

        return [C,] + item[1:]
    

class term_document_frequency(frequency_counter):

    table_name = "term_document_frequency"

    def __call__(self, item):

        text = item[0]   
        tokens = set(unicode(text).split())
        C = collections.Counter(tokens)

        # Add an empty string token to keep track of total documents
        C[""] += 1

        return [C,] + item[1:]
