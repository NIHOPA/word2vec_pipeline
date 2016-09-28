'''
TO DO: 

[ ] Add plots

'''

import os
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy

from data_utils import load_metacluster_data, load_document_vectors, load_SQL_data

def _compute_dispersion(X):
    return pdist(X, metric='cosine').mean()

def _compute_centroid_dist(X,cx):
    return cdist(X, [cx,], metric='cosine').mean()

def _compute_dendrogram_order(X, metric='cosine'):
    pairwise_dists = pdist(C,metric='cosine')
    linkage = hierarchy.linkage(pairwise_dists, method='single')
    dendro  = hierarchy.dendrogram(linkage, no_plot=True)
    return dendro["leaves"]    

if __name__ == "__main__" and __package__ is None:

    import simple_config
    config = simple_config.load("postprocessing")

    save_dest = config['output_data_directory']
    os.system('mkdir -p {}'.format(save_dest))

    SQL = load_SQL_data(config["master_columns"])

    MC = load_metacluster_data()
    C = MC["meta_centroids"]
    counts = collections.Counter(MC["meta_labels"])

    DV = load_document_vectors()

    # Build the results for the metaclusters
    labels = np.unique(MC["meta_labels"])

    ################################################################################
    print "Computing intra-document dispersion."
    
    V = DV["docv"]
    data = []
    for cx, cluster_id in tqdm(zip(C,labels)):
        idx = MC["meta_labels"]==cluster_id

        item = {}
        item["counts"] = idx.sum()
        item["intra_document_dispersion"] = _compute_dispersion(V[idx])
        item["avg_centroid_distance"] = _compute_centroid_dist(V[idx],cx)
        data.append(item)


    df = pd.DataFrame(data, index=labels)
    df.index.name = "cluster_id"
    df["word2vec_description"] = MC["describe_clusters"]
    df["dendrogram_order"] = _compute_dendrogram_order(C)

    cols = ["dendrogram_order", "counts",
            "avg_centroid_distance",
            "intra_document_dispersion",
            "word2vec_description"]

    df = df[cols].sort_values("dendrogram_order")

    f_csv = os.path.join(save_dest, "cluster_desc.csv")
    df.to_csv(f_csv, index_label="cluster_id")

    ################################################################################
    print "Computing master-label spreadsheets."
    cluster_lookup = dict(zip(df.index, df.dendrogram_order.values))
    SQL["cluster_id"] = MC["meta_labels"]
    SQL["dendrogram_order"] = -1

    for i,j in cluster_lookup.items():
        idx = SQL["cluster_id"]==i
        SQL.loc[idx, "dendrogram_order"] = j


    special_cols = ["_ref","cluster_id","dendrogram_order"]
    cols = [x for x in SQL.columns if x not in special_cols]
    SQL = SQL[special_cols + cols]
        
    f_csv = os.path.join(save_dest, "cluster_master_labels.csv")
    SQL.to_csv(f_csv,index=False)

