"""
Perform analysis on the document metaclusters created by the pipeline.
This will automatically provide labels to each cluster, by identifying
which words are most similar to the centroid of each cluster.

It also returns statistics on each cluster to determine the average
similarity of each document to the cluster centroid, and the average
similarity of each document to every other document in the cluster.
"""


import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy

import utils.data_utils as uds

import logging

logger = logging.getLogger(__name__)


def _compute_centroid_dist(X, cx):
    """
    Find the average distance of all documents in a cluster to its centroid
        X: a document vector
        cx: a list of cluster centroids
    Returns
        a float similarity value
    """
    return cdist(X, [cx], metric="cosine").mean()


def _compute_dispersion_matrix(X, labels):
    """
    Find the intra-document dispersion of every document in a cluster.

    Args:
        X: an numpy array of each document in a cluster's document vector
        labels: labels for each cluster
    Returns
        dist: a numpy array of the matrix of pairwise dispersion measures
              between each document in a cluster
    """

    n = len(np.unique(labels))
    dist = np.zeros((n, n))
    ITR = list(itertools.combinations_with_replacement(range(n), 2))
    for i, j in tqdm(ITR):

        if i == j:
            d = pdist(X[labels == i], metric="cosine")
        else:
            d = cdist(X[labels == i], X[labels == j], metric="cosine")
            # Only take upper diagonal (+diagonal elements)
            d = d[np.triu_indices(n=d.shape[0], m=d.shape[1], k=0)]

        dist[i, j] = dist[j, i] = d.mean()

    return dist


def analyze_metacluster_from_config(config):
    """
    Does analysis on metaclusters to return descriptive information and
    statistics.

    Args:
        config: a config file
    """

    score_method = config["metacluster"]["score_method"]
    config = config["analyze"]
    topn_words_returned = config["topn_words_returned"]

    save_dest = config["output_data_directory"]
    os.system("mkdir -p {}".format(save_dest))

    model = uds.load_w2vec()
    ORG = uds.load_ORG_data(config["master_columns"])

    MC = uds.load_metacluster_data()
    C = MC["meta_centroids"]

    DV = uds.load_document_vectors(score_method)

    # Fix any zero vectors with random ones
    dim = DV["docv"].shape[1]
    idx = np.where(np.linalg.norm(DV["docv"], axis=1) == 0)[0]
    for i in idx:
        vec = np.random.uniform(size=(dim,))
        vec /= np.linalg.norm(vec)
        DV["docv"][i] = vec

    # Build the results for the metaclusters
    labels = np.unique(MC["meta_labels"])

    if config["compute_dispersion"]:
        logger.info("Computing intra-document dispersion.")
        dist = _compute_dispersion_matrix(DV["docv"], MC["meta_labels"])

        # Compute the linkage and the order
        linkage = hierarchy.linkage(dist, method="average")
        d_idx = hierarchy.dendrogram(linkage, no_plot=True)["leaves"]

    else:
        # If dispersion is not calculated set d_idx to be the cluster index
        d_idx = np.sort(labels)

    #

    V = DV["docv"]
    data = []
    for cx, cluster_id in zip(C, labels):
        idx = MC["meta_labels"] == cluster_id

        item = {}
        item["counts"] = idx.sum()
        item["avg_centroid_distance"] = _compute_centroid_dist(V[idx], cx)

        if config["compute_dispersion"]:
            item["intra_document_dispersion"] = dist[cluster_id, cluster_id]
        else:
            item["intra_document_dispersion"] = -1

        # Compute closest words to the centroid
        desc = " ".join(
            list(
                zip(*model.wv.similar_by_vector(cx, topn=topn_words_returned))
            )[0]
        )
        item["word2vec_description"] = desc

        data.append(item)

    df = pd.DataFrame(data, index=labels)

    df.index.name = "cluster_id"
    df["dispersion_order"] = d_idx

    cols = [
        "dispersion_order",
        "counts",
        "avg_centroid_distance",
        "intra_document_dispersion",
        "word2vec_description",
    ]

    df = df[cols]

    f_csv = os.path.join(save_dest, "cluster_desc.csv")
    df.to_csv(f_csv, index_label="cluster_id")

    logger.info("Computing master-label spreadsheets.")
    cluster_lookup = dict(zip(df.index, df.dispersion_order.values))
    ORG["cluster_id"] = MC["meta_labels"]
    ORG["dispersion_order"] = -1

    for i, j in cluster_lookup.items():
        idx = ORG["cluster_id"] == i
        ORG.loc[idx, "dispersion_order"] = j

    special_cols = ["_ref", "cluster_id", "dispersion_order"]
    cols = [x for x in ORG.columns if x not in special_cols]

    ORG = ORG[special_cols + cols]

    if not config["compute_dispersion"]:
        del df["intra_document_dispersion"]
        del df["dispersion_order"]

    f_csv = os.path.join(save_dest, "cluster_master_labels.csv")
    ORG.to_csv(f_csv, index=False)

    print(df)  # Output the result to stdout


if __name__ == "__main__" and __package__ is None:

    import simple_config

    config = simple_config.load()
    analyze_metacluster_from_config(config)
