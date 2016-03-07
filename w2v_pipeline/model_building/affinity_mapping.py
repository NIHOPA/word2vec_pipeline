import collections, itertools, os
import numpy as np
import pandas as pd
import h5py

from gensim.models.word2vec import Word2Vec
from mapreduce import corpus_iterator

from sklearn.cluster import AffinityPropagation as cluster_clf
from scipy.spatial.distance import cdist
import tqdm

from scipy import sparse

damping = None
M = None

def compute_local_affinity(V):

    cluster_args = {"damping":0.95}
    cluster = cluster_clf(**cluster_args)

    #print "Clustering affinity document vectors", V.shape
    DV = cdist(V,V)
    z_labels = cluster.fit_predict(DV)

    #print "{} unique labels found".format(np.unique(z_labels).shape)
    return V,z_labels


def compute_affinity(item):

    text,f_idx,f_sql = item
    tokens = text.split()

    # Find out which tokens are defined
    valid_tokens = [w for w in tokens if w in M]
    
    local_counts = collections.Counter(valid_tokens)
    labels = np.array(list(set(valid_tokens)))

    token_clf_index = np.array([M.word2index[w]
                                for w in labels])

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

    data = {
        "token_clf_index":token_clf_index,
        "y_labels":y_labels,
    }
    
    '''
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
            "token_clf_index":token_clf_index[idx],
        }
    '''
    
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

        self.vocab_n = len(M.index2word)
    
        M.word2index = dict([(w,i) for w,i in
                             zip(M.index2word,range(self.vocab_n))])

        # Increment this as we find more clusters
        self.cluster_n = 0

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
                        
        # Save the size of the vocabulary
        self.h5["documents"].attrs["vocab_n"] = self.vocab_n        
        self.h5["documents"].attrs["cluster_n"] = self.cluster_n

        self.h5.close()

    def save(self, config, result):

        idx, f_sql, data = result
        
        key = os.path.basename(f_sql) + '_' + str(idx)

        g = self.h5.require_group("documents")
        g = self.h5["documents"].require_group(key)
        
        g["token_clf_index"] = data["token_clf_index"]
        g["y_labels"] = data["y_labels"]

        self.cluster_n += len(np.unique(data["y_labels"]))


class affinity_grouping(corpus_iterator):

    def __init__(self,*args,**kwargs):
        super(affinity_grouping, self).__init__(*args,**kwargs)

        # Set parallel option
        self._PARALLEL = kwargs["_PARALLEL"]

         # Load the model from disk
        self.h5 = h5py.File(kwargs["f_affinity"],'r+')

        # Capture the size of the vocabulary
        self.vocab_n = self.h5["documents"].attrs["vocab_n"]
        self.cluster_n = self.h5["documents"].attrs["cluster_n"]

        self.M = Word2Vec.load(kwargs["f_w2v"])

    def _iterator_mean_cluster_vectors(self):
                       
        g = self.h5["documents"]
        
        for k,name in enumerate(g):
            
            token_idx = g[name]["token_clf_index"][:]
            y_labels = g[name]["y_labels"][:]
            
            for i in np.unique(y_labels):
                idx = i==y_labels

                words = [self.M.index2word[word_idx]
                         for word_idx in token_idx[idx]]
                vec = np.array([self.M[w] for w in words])
                vec = vec.mean(axis=0)
                vec /= np.linalg.norm(vec)
                yield vec


    def iterator_batch_mean_vectors(self, batch_size=2000):

        V = []
        for x in self._iterator_mean_cluster_vectors():
            V.append(x)
            if len(V) == batch_size:
                yield np.array(V)
                V = []
        yield np.array(V)
            

    def compute(self, config):

        func = compute_local_affinity
        INPUT_ITR = self.iterator_batch_mean_vectors(500)
        
        if self._PARALLEL:
            import multiprocessing
            MP = multiprocessing.Pool()
            ITR = MP.imap(func, INPUT_ITR)
        else:
            ITR = itertools.imap(func, INPUT_ITR)

        Z = []

        for result in ITR:
            V,z_labels = result
            
            for i in np.unique(z_labels):
                z = V[i==z_labels].mean(axis=0)
                z /= np.linalg.norm(z)
                Z.append(z)

        print "Final affinity size", len(Z)
        print self.vocab_n, self.cluster_n

        # Need to recluster this again to make it smaller!
        # 85376 -> 7543 -> ....


    '''

            continue
            
            import pylab as plt

            DV = cdist(V,V)
            print DV.shape

            sort_idx = np.argsort(z_labels)
            z_labels = z_labels[sort_idx]
            
            import seaborn as sns
            sns.heatmap(DV, xticklabels=False, yticklabels=False)
            
            DV = DV[sort_idx,:][:,sort_idx]
            sns.plt.figure()
            sns.heatmap(DV, xticklabels=False, yticklabels=False)
            plt.show()
            print z_labels.max()
            print z_labels
        

        exit()

        

        
        X = []
        for k,name in enumerate(g):
            token_idx = g[name]["token_clf_index"][:]
            y_labels = g[name]["y_labels"][:]

            for i in np.unique(y_labels):
                x = np.zeros(shape=(self.vocab_n,))
                idx = y_labels == i
                size = idx.sum()

                if size>2 and size<10:
                    x[ token_idx[idx] ] = True

                    # Aggregate this
                    X.append(x)
            #print k
            if k>1000: break

        X = np.array(X)
        print len(X)
        print X
        print X.sum()
        print X.shape
        exit()
        
        #size = np.hstack([g[name]["size"] for name in g])
        #mu   = np.hstack([g[name]["mu"] for name in g])
        print self.vocab_n

        print g
        exit()

        print mu
        import seaborn as sns
        sns.distplot(mu)
        sns.plt.show()
        
        
        idx  = (size>=5) * (size<=10) * (mu < 0.65)
        #idx  = (size>=5) * (size<=10) * (mu < 0.5)

        V = np.vstack([g[name]["V"][:] for name in g])
        V = V[idx]

        print "Clustering affinity", V.shape
        
        cluster_args = {
            "n_clusters":50,
            "n_jobs":-1,
            "verbose":0,
        }
        cluster = cluster_clf(**cluster_args)
        y_labels = cluster.fit_predict(V)

        affinity_V = []

        for i in sorted(np.unique(y_labels)):
            v  = V[i==y_labels].mean(axis=0)
            v /= np.linalg.norm(v)
            affinity_V.append(v)

        affinity_V = np.array(affinity_V)

        if "affinity_vectors" in self.h5:
            del self.h5["affinity_vectors"]
            
        self.h5["affinity_vectors"] = affinity_V
    '''
    
    def save(self, config, result):
        pass
