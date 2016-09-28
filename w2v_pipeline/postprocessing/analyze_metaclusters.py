'''
TO DO: 

[ ] Add dendrogram_id
[ ] Add master labels

'''

import os
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist

from data_utils import load_metacluster_data, load_document_vectors

def _compute_dispersion(X):
    return pdist(X, metric='cosine').mean()

def _compute_centroid_dist(X,cx):
    return cdist(X, [cx,], metric='cosine').mean()

if __name__ == "__main__" and __package__ is None:

    import simple_config
    config = simple_config.load("postprocessing")

    save_dest = config['output_data_directory']
    os.system('mkdir -p {}'.format(save_dest))

    DV = load_document_vectors()
    MC = load_metacluster_data()

    C = MC["meta_centroids"]
    counts = collections.Counter(MC["meta_labels"])

    # Build the results for the metaclusters
    labels = np.unique(MC["meta_labels"])
    
    # Load the document vectors
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

    f_csv = os.path.join(save_dest, "cluster_desc.csv")
    df.to_csv(f_csv, index_label="cluster_id")
