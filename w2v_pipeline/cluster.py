import numpy as np
import h5py
import os
import itertools
import collections

import clustering.similarity as CSIM
from utils.os_utils import mkdir
from unidecode import unidecode
from sklearn.manifold import TSNE


class cluster_object(object):

    '''
    Helper class to represent all the constitute parts of a clustering
    '''

    def __init__(self, h5,
                 document_score_method,
                 cluster_method,
                 target_column):

        self.name_doc = document_score_method
        self.name_cluster = cluster_method

        g = h5[document_score_method]

        self.labels = g["clustering"][cluster_method][:]
        self.words = g["nearby_words"][cluster_method][:]
        self.T = g["tSNE"][:]
        self.S = g["similarity"][:]
        self.X = load_document_vectors(target_column)
        h5.close()

        assert(self.X.shape[0] == self.T.shape[0] == self.S.shape[0])

        self._label_iter = np.sort(np.unique(self.labels))
        self.cluster_n = self._label_iter.size

    def reorder(self, idx):
        self.X = self.X[idx]
        self.S = self.S[idx][:, idx]
        self.labels = self.labels[idx]
        self.T = self.T[idx]

    def __len__(self):
        return self.X.shape[0]

    def sort_labels(self):
        # Reorder the data so the data is the assigned cluster
        idx = np.argsort(self.labels)
        self.reorder(idx)

    def sort_intra(self):
        master_idx = np.arange(len(self))

        for i in self._label_iter:
            cidx = self.labels == i
            Z = self.X[cidx]
            zmu = Z.sum(axis=0)
            zmu /= np.linalg.norm(zmu)

            dist = Z.dot(zmu)
            dist_idx = np.argsort(dist)
            master_idx[cidx] = master_idx[cidx][dist_idx]

        self.reorder(master_idx)


def close_words(W, X, labels, top_n=6):
    '''
    Find words that are close to each label.
    W is a gensim.word2vec
    X is the document vectors.
    labels are predetermined cluster labels.
    '''

    L = []
    for label in np.unique(labels):
        label_idx = labels == label
        mu = X[label_idx].mean(axis=0)

        dist = W.syn0.dot(mu)
        idx = np.argsort(dist)[::-1][:top_n]
        words = [W.index2word[i] for i in idx]
        L.append(' '.join(words))

    # Map unicode to simple ASCII
    L = map(unidecode, L)

    # Remove _PHRASE
    L = map(lambda x: x.replace('PHRASE_', ''), L)

    return L


def load_document_vectors(target_column):

    config_score = simple_config.load("score")

    f_h5 = os.path.join(
        config_score["output_data_directory"],
        config_score["document_scores"]["f_db"],
    )

    with h5py.File(f_h5, 'r') as h5_score:
        assert(method in h5_score)
        g = h5_score[method]

        print "Loading the document scores", g
        X = g["V"][:]

    return X


if __name__ == "__main__":

    import simple_config

    config = simple_config.load("cluster")
    output_dir = config["output_data_directory"]
    mkdir(output_dir)

    method = config['score_method']
    target_column = config['score_column']

    f_sim = os.path.join(output_dir, config["f_cluster"])

    if config.as_bool("_FORCE"):
        try:
            os.remove(f_sim)
        except:
            pass

    if not os.path.exists(f_sim):
        h5_sim = h5py.File(f_sim, 'w')
        h5_sim.close()

    h5_sim = h5py.File(f_sim, 'r+')
    group = h5_sim.require_group(method)
    S = None
    W = None

    # Load the document scores
    X = load_document_vectors(target_column)

    if "similarity" not in group:

        # Compute and save the similarity matrix
        print "Computing the similarity matrix"

        # Save the similarity matrix
        S = CSIM.compute_document_similarity(X)
        group["similarity"] = S

    if "tSNE" not in group:
        # Compute the tSNE
        print "Computing tSNE for {}".format(method)

        if S is None:
            S = group["similarity"][:]
        local_embed = TSNE(n_components=2,
                           verbose=1,
                           method='exact',
                           metric='precomputed')

        # tSNE expect distances not similarities
        group["tSNE"] = local_embed.fit_transform(1 - S)

    group.require_group("clustering")
    group.require_group("nearby_words")

    for name in config["clustering_commands"]:

        if name not in group["clustering"]:
            # Only load the similarity matrix if needed
            if S is None:
                S = group["similarity"][:]
            if W is None:
                W = CSIM.load_embeddings()

            print "Clustering {} {}".format(method, name)

            func = getattr(CSIM, name)
            labels = func(S, X, config[name])

            assert(labels.size == X.shape[0])

            if name in group["clustering"]:
                del group["clustering"][name]
            group["clustering"][name] = labels

            L = close_words(W, X, labels)
            if name in group["nearby_words"]:
                del group["nearby_words"][name]
            group["nearby_words"].create_dataset(name, data=L, dtype='S200')

    # Load the cluster object
    document_score_method = method
    cluster_method = "spectral_clustering"

    # cluster_method = "hdbscan_clustering"

    C = cluster_object(h5_sim,
                       document_score_method,
                       cluster_method,
                       target_column)

    # Sort by labels first
    C.sort_labels()

    # Reorder the intra-clusters by closest to centroid
    C.sort_intra()

    # Plot the heatmap
    import pandas as pd
    import seaborn as sns
    plt = sns.plt

    fig = plt.figure(figsize=(9, 9))
    print "Plotting tSNE"
    colors = sns.color_palette("hls", C.cluster_n)

    for i in C._label_iter:
        x, y = zip(*C.T[C.labels == i])
        label = 'cluster {}, {}'.format(i, C.words[i])

        plt.scatter(x, y, color=colors[i], label=label)
    plt.title("tSNE doc:{} plot".format(C.name_doc), fontsize=16)
    plt.legend(loc=1, fontsize=12)
    plt.tight_layout()

    fig = plt.figure(figsize=(12, 12))
    print "Plotting heatmap"
    df = pd.DataFrame(C.S, index=C.labels, columns=C.labels)
    labeltick = int(len(df) / 50)

    sns.heatmap(df, cbar=False, xticklabels=labeltick, yticklabels=labeltick)
    plt.title("heatmap doc:{} clustering:{}".format(
        C.name_doc, C.name_cluster), fontsize=16)
    plt.tight_layout()
    # plt.savefig("clustering_{}.png".format(n_clusters))
    plt.show()
