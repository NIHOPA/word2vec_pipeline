import sqlite3, os
import pandas as pd
import numpy as np

def findall_substr(sub, string):
    """
    >>> text = "Allowed Hello Hollow"
    >>> tuple(findall('ll', text))
    (1, 10, 16)
    http://stackoverflow.com/a/3874760/249341
    """
    index = 0 - len(sub)
    try:
        while True:
            index = string.index(sub, index + len(sub))
            yield index
    except ValueError:
        pass
        
class replace_from_dictionary(object):
    '''
    DOCSTRING: TO WRITE.
    '''
    
    def __init__(self, f_dict):

        if not os.path.exists(f_dict):
            msg = "Can't find dictionary {}".format(f_dict)
            raise IOError(msg)

        df = pd.read_csv(f_dict)
        items = df["SYNONYM"].str.lower(), df["replace_token"]
        self.X = dict(zip(*items))
                                       
    def __call__(self,org_doc):

        doc = org_doc
        ldoc = doc.lower()

        # Identify which phrases were used
        keywords = [key for key in self.X if key in ldoc]

        # Loop over the keywords and replace them one-by-one.
        # This is inefficient, but less error prone.
        for word in keywords:
            while word in ldoc:
                idx = ldoc.index(word)
                doc = doc[:idx] + self.X[word] + doc[idx+len(word):]
                ldoc = doc.lower()
        print doc
        exit()
        return doc
