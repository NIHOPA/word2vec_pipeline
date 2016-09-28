import h5py, sqlite3, os
import numpy as np

# Required for import from previous path (may fix someday)
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import simple_config

def load_metacluster_data(*args):

    config_metacluster = simple_config.load("metacluster")

    f_h5 = os.path.join(
        config_metacluster["output_data_directory"],
        config_metacluster["f_centroids"])

    data = {}
    
    with h5py.File(f_h5,'r') as h5:
        if not args:
            args = h5.keys()
            
        for key in args:
            if key not in h5:
                raise ValueError("{} not found in {}".format(key, f_h5))

        for key in args:
            data[key] = h5[key][:]

    return data


def load_document_vectors():
    config_score = simple_config.load("score")
    config_MC = simple_config.load("metacluster")

    score_method = config_MC['score_method']
    text_column  = config_MC['score_column']
    
    f_h5 = os.path.join(
        config_score["output_data_directory"],
        config_score['document_scores']["f_db"],
    )

    with h5py.File(f_h5,'r') as h5:
        g = h5[score_method][text_column]
        corpus_keys = g.keys()

        # Load the _refs
        _refs = np.hstack([g[key]["_ref"][:] for key in corpus_keys])
        
        # Require the _refs to be in order as a sanity check
        if not (np.sort(_refs) == _refs).all():
            msg = "WARNING, data out of sort order from _refs"
            raise ValueError(msg)
        
        docv = np.vstack([g[k]["V"][:] for k in corpus_keys])

        return {
            "docv" : docv,
            "_refs": _refs
        }
