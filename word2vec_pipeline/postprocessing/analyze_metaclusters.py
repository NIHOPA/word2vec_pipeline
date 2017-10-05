import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy

import utils.data_utils as uds

def _compute_centroid_dist(X, cx):
    return cdist(X, [cx, ], metric='cosine').mean()

def _compute_dispersion_matrix(X, labels):
    n = len(np.unique(labels))
    dist = np.zeros((n, n))
    ITR = list(itertools.combinations_with_replacement(range(n), 2))
    for i, j in tqdm(ITR):

        if i == j:
            d = pdist(X[labels == i], metric='cosine')
        else:
            d = cdist(X[labels == i], X[labels == j], metric='cosine')
            # Only take upper diagonal (+diagonal elements)
            d = d[np.triu_indices(n=d.shape[0], m=d.shape[1], k=0)]

        dist[i, j] = dist[j, i] = d.mean()

    return dist


def analyze_metacluster_from_config(config):

    score_method = config["metacluster"]["score_method"]
    config = config["postprocessing"]

    save_dest = config['output_data_directory']
    os.system('mkdir -p {}'.format(save_dest))

    model = uds.load_w2vec()
    ORG = uds.load_ORG_data(config["master_columns"])

    MC = uds.load_metacluster_data()
    C = MC["meta_centroids"]

    DV = uds.load_document_vectors(score_method)

    # Fix any zero vectors with random ones
    dim = DV["docv"].shape[1]
    idx = np.where(np.linalg.norm(DV["docv"],axis=1)==0)[0]
    for i in idx:
        vec = np.random.uniform(size=(dim,))
        vec /= np.linalg.norm(vec)
        DV["docv"][i] = vec

    # Build the results for the metaclusters
    labels = np.unique(MC["meta_labels"])

    print("Computing intra-document dispersion.")
    dist = _compute_dispersion_matrix(DV["docv"], MC["meta_labels"])

    # Compute the linkage and the order
    linkage = hierarchy.linkage(dist, method='average')
    d_idx = hierarchy.dendrogram(linkage, no_plot=True)["leaves"]

    #

    V = DV["docv"]
    data = []
    for cx, cluster_id in zip(C, labels):
        idx = MC["meta_labels"] == cluster_id

        item = {}
        item["counts"] = idx.sum()
        item["intra_document_dispersion"] = dist[cluster_id, cluster_id]
        item["avg_centroid_distance"] = _compute_centroid_dist(V[idx], cx)

        # Compute closest words to the centroid
        desc = ' '.join(zip(*model.wv.similar_by_vector(cx))[0])
        item["word2vec_description"] = desc
    
        data.append(item)

    df = pd.DataFrame(data, index=labels)

    df.index.name = "cluster_id"
    df["dendrogram_order"] = d_idx

    cols = [
        "dendrogram_order",
        "counts",
        "avg_centroid_distance",
        "intra_document_dispersion",
        "word2vec_description"
    ]

    df = df[cols].sort_values("dendrogram_order")

    f_csv = os.path.join(save_dest, "cluster_desc.csv")
    df.to_csv(f_csv, index_label="cluster_id")

    #

    print("Computing master-label spreadsheets.")
    cluster_lookup = dict(zip(df.index, df.dendrogram_order.values))
    ORG["cluster_id"] = MC["meta_labels"]
    ORG["dendrogram_order"] = -1

    for i, j in cluster_lookup.items():
        idx = ORG["cluster_id"] == i
        ORG.loc[idx, "dendrogram_order"] = j

    special_cols = ["_ref", "cluster_id", "dendrogram_order"]
    cols = [x for x in ORG.columns if x not in special_cols]

    ORG = ORG[special_cols + cols]

    f_csv = os.path.join(save_dest, "cluster_master_labels.csv")
    ORG.to_csv(f_csv, index=False)

    #

    df["cluster_id"] = df.index
    df = df.sort_values("cluster_id")
    print(df)
    f_h5_save = os.path.join(save_dest, "cluster_dispersion.h5")
    
    with h5py.File(f_h5_save, 'w') as h5_save:
        h5_save["dispersion"] = dist
        h5_save["cluster_id"] = df.cluster_id
        h5_save["counts"] = df.counts
        h5_save["dendrogram_order"] = df.dendrogram_order
        h5_save["linkage"] = linkage


if __name__ == "__main__" and __package__ is None:

    import simple_config
    config = simple_config.load()
    analyze_metacluster_from_config(config)
