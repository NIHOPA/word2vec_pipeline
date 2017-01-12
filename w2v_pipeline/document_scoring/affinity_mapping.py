import collections
import itertools
import os
import ast
import numpy as np
import pandas as pd
import h5py

from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator

from sklearn.cluster import AffinityPropagation as cluster_clf
from scipy.spatial.distance import cdist
import tqdm

from scipy import sparse
from sklearn.decomposition import SparseCoder

from utils.parallel_utils import jobmap

damping = None
M = None
sparse_coder = None


def compute_local_affinity(V):
    global damping

    cluster_args = {"damping": damping}
    cluster = cluster_clf(**cluster_args)

    DV = cdist(V, V, metric='cosine')
    z_labels = cluster.fit_predict(DV)

    # print "{} unique labels found".format(np.unique(z_labels).shape)
    return V, z_labels


def compute_affinity(item):

    text, f_idx, table_name, f_sql = item
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

    V = np.array([M[w] for w in labels])
    DV = cdist(V, V, metric='cosine')

    # Values are sometimes "slightly" less than zero due to rounding
    DV[DV < 0] = 0

    cluster_args = {"damping": damping}
    cluster = cluster_clf(**cluster_args)

    y_labels = cluster.fit_predict(DV)

    data = {}

    data = {
        "token_clf_index": token_clf_index,
        "y_labels": y_labels,
    }

    return f_idx, f_sql, data


class affinity_mapping(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(affinity_mapping, self).__init__(*args, **kwargs)

         # Load the model from disk
        self.M = Word2Vec.load(kwargs["f_w2v"])
        self.shape = self.M.syn0.shape

        # Set parallel option
        self._PARALLEL = ast.literal_eval(kwargs["_PARALLEL"])

        self.damping = float(kwargs["damping"])

        if not os.path.exists(kwargs["f_affinity"]):
            h5 = h5py.File(kwargs["f_affinity"], 'w')
            h5.close()

        self.h5 = h5py.File(kwargs["f_affinity"], 'r+')

        global damping, M

        damping = self.damping
        M = self.M

        self.vocab_n = len(M.index2word)

        M.word2index = dict([(w, i) for w, i in
                             zip(M.index2word, range(self.vocab_n))])

        # Increment this as we find more clusters
        self.cluster_n = 0

    def compute(self, config):

        func = compute_affinity
        ITR = jobmap(func, self, self.PARALLEL)
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

        for key in ["token_clf_index", "y_labels"]:
            if key in g:
                del g[key]
            g[key] = data[key]

        self.cluster_n += len(np.unique(data["y_labels"]))

#


class affinity_grouping(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(affinity_grouping, self).__init__(*args, **kwargs)

        # Set parallel option
        self._PARALLEL = ast.literal_eval(kwargs["_PARALLEL"])

         # Load the model from disk
        self.h5 = h5py.File(kwargs["f_affinity"], 'r+')

        # Capture the size of the vocabulary
        self.vocab_n = self.h5["documents"].attrs["vocab_n"]
        self.cluster_n = self.h5["documents"].attrs["cluster_n"]

        self.batch_size = int(kwargs["batch_size"])

        global damping
        damping = float(kwargs["damping"])

        self.M = Word2Vec.load(kwargs["f_w2v"])

    def _iterator_mean_cluster_vectors(self):

        g = self.h5["documents"]

        for k, name in enumerate(g):

            token_idx = g[name]["token_clf_index"][:]
            y_labels = g[name]["y_labels"][:]

            for i in np.unique(y_labels):
                idx = i == y_labels

                words = [self.M.index2word[word_idx]
                         for word_idx in token_idx[idx]]
                vec = np.array([self.M[w] for w in words])
                vec = vec.mean(axis=0)
                vec /= np.linalg.norm(vec)
                yield vec

    def iterator_batch(self, ITR):

        V = []
        for x in ITR:
            V.append(x)
            if len(V) == self.batch_size:
                yield np.array(V)
                V = []
        yield np.array(V)

    def cluster_affinity_states(self, INPUT_ITR, size=0):

        func = compute_local_affinity
        ITR = jobmap(func, INPUT_ITR, self.PARALLEL)

        Z = []

        pbar = tqdm.tqdm(total=size // self.batch_size)

        for result in ITR:
            V, z_labels = result

            for i in np.unique(z_labels):
                z = V[i == z_labels].mean(axis=0)
                z /= np.linalg.norm(z)
                Z.append(z)

            pbar.update()

        pbar.close()

        return np.array(Z)

    def compute(self, config):

        INPUT_ITR = self.iterator_batch(self._iterator_mean_cluster_vectors())
        Z = self.cluster_affinity_states(INPUT_ITR, size=self.cluster_n)

        print "Initial affinity grouping", Z.shape
        # print self.vocab_n, self.cluster_n

        INPUT_ITR = self.iterator_batch(Z)
        Z2 = self.cluster_affinity_states(INPUT_ITR, size=len(Z))

        print "Final affinity size", len(Z2)
        self.save(config, Z2)

        '''
        import seaborn as sns
        plt = sns.plt
        DZ2 = cdist(Z2,Z2,metric='cosine')
        sns.heatmap(DZ2,xticklabels=False, yticklabels=False,linewidths=0)
        sns.plt.figure()
        #plt.show()

        DZ = cdist(Z,Z,metric='cosine')
        sns.heatmap(DZ,xticklabels=False, yticklabels=False,linewidths=0)
        #sns.plt.figure()
        sns.plt.show()
        '''

        self.h5.close()

    def save(self, config, result):

        name = "clustered_affinity"

        if name in self.h5:
            del self.h5[name]

        self.h5[name] = result

#


def compute_document_affinity(item):
    global M, sparse_coder

    f_idx, f_sql, data = compute_affinity(item)

    token_clf_index = data["token_clf_index"]
    y_labels = data["y_labels"]

    cluster_vecs = []
    cluster_size = []

    for i in np.unique(y_labels):
        idx = i == y_labels

        words = [M.index2word[word_idx]
                 for word_idx in token_clf_index[idx]]
        vec = np.array([M[w] for w in words])
        vec = vec.mean(axis=0)
        vec /= np.linalg.norm(vec)

        cluster_vecs.append(vec)
        cluster_size.append(len(words))

    cluster_vecs = np.array(cluster_vecs)
    cluster_size = np.array(cluster_size)

    # Compute the sparse coding
    coding = sparse_coder.transform(cluster_vecs)

    # Scale the coding based of the size of the cluster
    coding = (coding.T * cluster_size).T

    # Project all the clusters down and normalize this vector
    doc_rep = coding.sum(axis=0)
    doc_rep /= np.linalg.norm(doc_rep)

    return (doc_rep, f_idx, f_sql)


class affinity_scoring(affinity_mapping):

    def __init__(self, *args, **kwargs):
        super(affinity_scoring, self).__init__(*args, **kwargs)
        self.A = self.h5["clustered_affinity"][:]

        global damping
        damping = float(kwargs["affinity_grouping"]["damping"])

        global sparse_coder
        nzero = kwargs["n_nonzero_coeffs"]

        sparse_coder = SparseCoder(self.A, split_sign=False,
                                   transform_n_nonzero_coefs=nzero)

    def compute(self, config):

        func = compute_document_affinity
        ITR = jobmap(func, self, self.PARALLEL)

        doc_data = []

        print "Computing document affinity scoring"
        for result in ITR:
            doc_data.append(result)

        df = pd.DataFrame(data=doc_data,
                          columns=["V", "idx", "f_sql"])

        self.save(config, df)

    def save(self, config, df):

        method = "affinity"

        print "Saving the scored documents"
        f_db = config["document_scores"]["f_db"]

        # Create the h5 file if it doesn't exist
        if not os.path.exists(f_db):
            h5 = h5py.File(f_db, 'w')
        else:
            h5 = h5py.File(f_db, 'r+')

        for key, data_group in df.groupby("f_sql"):

            # Save into the group of the base file name
            name = '.'.join(os.path.basename(key).split('.')[:-1])

            g = h5.require_group(method)
            V = np.array(data_group["V"].tolist())
            print "Saving", name, method, V.shape

            if name in g:
                del g[name]

            g.create_dataset(name,
                             data=V,
                             compression='gzip')

        h5.close()
