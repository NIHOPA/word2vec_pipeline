import h5py
import os
import pandas as pd
import numpy as np
import joblib

import simple_config
from os_utils import grab_files

"""
Utility file to assist in loading data saved as part of the pipeline, including .csv files, h5 files, amd gensim
w2v models. There are also functions to extract specific data from these files.
"""

def load_h5_file(f_h5, *args):
    '''
    Generically  loads a h5 files top level data structures (assumes
    no nesting). If *args is specified, only the *args subset will be loaded.
    '''
    data = {}

    with h5py.File(f_h5, 'r') as h5:
        if not args:
            args = h5.keys()

        for key in args:
            if key not in h5:
                raise ValueError("{} not found in {}".format(key, f_h5))

        for key in args:
            data[key] = h5[key][:]

    return data


def touch_h5(f_db):
    # Create the h5 file if it doesn't exist
    if not os.path.exists(f_db):
        h5 = h5py.File(f_db, 'w')
    else:
        h5 = h5py.File(f_db, 'r+')
    return h5


def load_dispersion_data():
    print("Loading dispersion data")

    config_post = simple_config().load["postprocessing"]

    f_h5 = os.path.join(
        config_post["output_data_directory"],
        "cluster_dispersion.h5")

    return load_h5_file(f_h5)

def simple_CSV_read(f, cols):
    try:
        dfx = pd.read_csv(f, usecols=cols)
    except ValueError:
        csv_cols = pd.read_csv(f, nrows=0).columns
        msg = "Columns requested {}, do not match columns in input csv {}"
        raise ValueError(msg.format(cols, csv_cols))
    return dfx


def load_ORG_data(extra_columns=None):
    print("Loading import data")

    cols = ["_ref", ]

    if extra_columns is not None:
        cols += extra_columns

    config = simple_config.load()
    config_import = config["import_data"]

    CORES = -1 if config["_PARALLEL"] else 1

    # Load the input columns
    F_CSV = grab_files("*.csv", config_import["output_data_directory"])

    with joblib.Parallel(CORES) as MP:
        func = joblib.delayed(simple_CSV_read)
        data = MP(func(x, cols) for x in F_CSV)

    # Require the _refs to be in order
    df = pd.concat(data).sort_values('_ref').set_index('_ref')

    # Use _ref as an index, but keep it as a row
    df['_ref'] = df.index

    return df


def load_metacluster_data(*args):

    config_metacluster = simple_config.load()["metacluster"]

    f_h5 = os.path.join(
        config_metacluster["output_data_directory"],
        config_metacluster["f_centroids"])

    return load_h5_file(f_h5, *args)


def get_score_methods():
    config_score = simple_config.load()["score"]

    f_h5 = os.path.join(
        config_score["output_data_directory"],
        config_score['document_scores']["f_db"],
    )

    with h5py.File(f_h5, 'r') as h5:
        return h5.keys()

def load_document_vectors(score_method, use_reduced=False):
    config_score = simple_config.load()["score"]

    f_h5 = os.path.join(
        config_score["output_data_directory"],
        config_score['document_scores']["f_db"],
    )

    with h5py.File(f_h5, 'r') as h5:

        assert(score_method in h5)
        g = h5[score_method]

        _refs = np.hstack([g[k]["_ref"][:] for k in g.keys()])
        
        vector_key = "VX" if use_reduced else "V"
        X = np.vstack([g[k][vector_key][:] for k in g.keys()])

        assert(X.shape[0] == _refs.size)

        # Sort to the proper order
        sort_idx = np.argsort(_refs)
        _refs = _refs[sort_idx]
        X = np.vstack(X)[sort_idx]
        
    return {
        "docv": X,
        "_refs": _refs
    }


def load_w2vec(config=None):
    import gensim.models.word2vec as W2V

    if config is None:
        config = simple_config.load()

    config_embed = config["embedding"]

    f_w2v = os.path.join(
        config_embed["output_data_directory"],
        config_embed["w2v_embedding"]["f_db"],
    )

    return W2V.Word2Vec.load(f_w2v)
