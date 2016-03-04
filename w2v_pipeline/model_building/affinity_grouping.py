import collections, itertools, os
import numpy as np
import pandas as pd
import h5py

from gensim.models.word2vec import Word2Vec
from mapreduce import corpus_iterator

from sklearn.cluster import AffinityPropagation as cluster_clf
from scipy.spatial.distance import cdist
import tqdm

damping = None
M = None

def compute_affinity(item):

    text,f_idx,f_sql = item
    tokens = text.split()

    # Find out which tokens are defined
    valid_tokens = [w for w in tokens if w in M]
    local_counts = collections.Counter(valid_tokens)
    labels = np.array(list(set(valid_tokens)))

    if not labels.size:
        msg = "Document has no valid tokens! This is problem."
        raise ValueError(msg)

    V  = np.array([M[w] for w in labels])
    DV = cdist(V,V,metric='cosine')

    # Values are sometimes "slightly" less than zero due to rounding
    DV[DV<0] = 0

    cluster_args = {"damping":damping}
    cluster = cluster_clf(**cluster_args)

    y_labels = cluster.fit_predict(DV)

    data = {}

    for i in sorted(np.unique(y_labels)):
        idx = y_labels==i

        distance_block = DV[idx,:][:,idx]
        vector_block = V[idx,:]


        average_vector  = vector_block.sum(axis=0)
        average_vector /= np.linalg.norm(average_vector)

        data[i] = {
            "average_vector":average_vector,
            "intra_mean":distance_block.mean(),
            "intra_std" :distance_block.std(),
            "size":vector_block.shape[0],
        }

    return f_idx, f_sql, data

class affinity_mapping(corpus_iterator):

    def __init__(self,*args,**kwargs):
        super(affinity_mapping, self).__init__(*args,**kwargs)

         # Load the model from disk
        self.M = Word2Vec.load(kwargs["f_w2v"])       
        self.shape = self.M.syn0.shape
        
        # Set parallel option
        self._PARALLEL = kwargs["_PARALLEL"]

        # Set parallel option
        self.damping = float(kwargs["damping"])
        self.h5 = h5py.File(kwargs["f_db"],'w')

        global damping, M

        damping = self.damping
        M = self.M

    def compute_affinity(self, item):
        
        text,f_idx,f_sql = item
        tokens = text.split()

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.M]
        local_counts = collections.Counter(valid_tokens)
        labels = np.array(list(set(valid_tokens)))

        if not labels.size:
            msg = "Document has no valid tokens! This is problem."
            raise ValueError(msg)

        V  = np.array([self.M[w] for w in labels])
        DV = cdist(V,V,metric='cosine')

        # Values are sometimes "slightly" less than zero due to rounding
        DV[DV<0] = 0

        cluster_args = {"damping":self.damping}
        cluster = cluster_clf(**cluster_args)

        y_labels = cluster.fit_predict(DV)
       
        data = {}
        
        for i in sorted(np.unique(y_labels)):
            idx = y_labels==i
            
            distance_block = DV[idx,:][:,idx]
            vector_block = V[idx,:]


            average_vector  = vector_block.sum(axis=0)
            average_vector /= np.linalg.norm(average_vector)

            data[i] = {
                "average_vector":average_vector,
                "intra_mean":distance_block.mean(),
                "intra_std" :distance_block.std(),
                "size":vector_block.shape[0],
            }

        return f_idx, f_sql, data


    def compute(self, config):

        func = compute_affinity
        
        if self._PARALLEL:
            import multiprocessing
            MP = multiprocessing.Pool()
            ITR = MP.imap(func, self)
        else:
            ITR = itertools.imap(func, self)

        print "Computing affinity propagation"

        for result in tqdm.tqdm(ITR):
            self.save(config, result)

        self.h5.close()

    def save(self, config, result):

        idx, f_sql, data = result
        
        key = os.path.basename(f_sql) + '_' + str(idx)
        g = self.h5.require_group(key)

        I = sorted(data.keys())
        g["mu"] = [data[i]["intra_mean"] for i in I]
        g["std"] = [data[i]["intra_std"] for i in I]
        g["size"] = [data[i]["size"] for i in I]
        g["V"] = [data[i]["average_vector"] for i in I]
        

