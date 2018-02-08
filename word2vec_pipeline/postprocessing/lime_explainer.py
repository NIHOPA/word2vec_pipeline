"""
Runs LIME the inter document metaclusters created by the pipeline.
For each pair of clusters connected by some threshold, LIME is run
against those clusters and the results are saved.
"""

'''
import os
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist
from scipy.cluster import hierarchy

'''
import utils.db_utils as udb
import utils.data_utils as uds
import utils.os_utils as uos

import document_scoring as ds

import numpy as np
import collections

import sklearn.ensemble
import sklearn.preprocessing
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
import pandas as pd

from lime.lime_text import LimeTextExplainer
import os
import joblib
from tqdm import tqdm

    
M = ds.score_unique()
explainer = LimeTextExplainer()

def _vectorizer(text_blocks):
    v = np.array([M(x) for x in text_blocks])
    return v

vectorizer = sklearn.preprocessing.FunctionTransformer(
    _vectorizer, validate=False)

def _explain_text(text, P, num_features):
    exp = explainer.explain_instance(
        text,
        P.predict_proba,
        num_features=num_features, 
    )
    item = collections.Counter()
    for k, v in exp.as_list():
        item[k] += v
    return item

def _compute_LIME(TEXT, Y, n_estimators, n_lime_features):

    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
    )

    P = Pipeline([
        ('word2vec', vectorizer),
        ('randomforests', clf),
    ])

    #class_names = LE.classes_
    explainer = LimeTextExplainer()

    P.fit(TEXT, Y)

    func = joblib.delayed(_explain_text)
    ITR = tqdm(TEXT)
    data = collections.Counter()
    
    with joblib.Parallel(-1) as MP:
        for c in MP(func(x, P, n_lime_features) for x in ITR):
            data.update(c)

    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df = df.sort_values(0)
    df.columns = ["word", "score"]
    df.score /= len(TEXT)
    
    return df
    

def _select_even_subset(Y, n_samples):
    # Select a subset for an even amount
    seq = np.arange(len(Y))
    
    idx0 = np.random.choice(seq[Y==0], size=n_samples, replace=False)
    idx1 = np.random.choice(seq[Y==1], size=n_samples, replace=False)
    new_idx = np.hstack([idx0,idx1])
    np.random.shuffle(new_idx)
    return new_idx



def explain_metaclusters(config):
    save_dest = config["postprocessing"]["output_data_directory"]
    uos.mkdir(save_dest)
    
    args = config["postprocessing"]["LIME_explainer"]

    f_csv_out = os.path.join(save_dest, "cluster_LIME.csv")
    
    data = uds.load_metacluster_data()
    centroids = data["meta_centroids"]
    labels = data["meta_labels"]

    # Find out which centroids are close, and find their index locations
    C = cdist(centroids, centroids, metric='cosine')
    cidx = np.where(C>float(args['metacluster_cosine_minsim']))

    n_lime_samples = int(args["n_lime_samples"])
    n_lime_features = int(args["n_lime_features"])
    n_estimators = int(args["n_estimators"])

    INPUT_ITR = udb.text_iterator()
    ALL_TEXT = np.array([row['text'] for row in INPUT_ITR])

    data = []
    for i,j in zip(*cidx):
        # Only take the upper diagonal
        if i>=j: continue

        print("Computing LIME for clusters {} and {}".format(i,j))

        labels_i = labels==i
        labels_j = labels==j       
        idx = labels_i|labels_j

        LE = sklearn.preprocessing.LabelEncoder()
        Y = LE.fit_transform(labels[idx])

        n_samples = min(labels_i.sum(), labels_j.sum(), n_lime_samples)

        new_idx = _select_even_subset(Y, n_samples)
        Y = Y[new_idx]
        TEXT = ALL_TEXT[idx][new_idx]

        df = _compute_LIME(TEXT, Y, n_estimators, n_lime_features)

        # Remove words that contributes < 0.5%
        df.score /= np.abs(df.score).sum()
        df = df[np.abs(df.score)>0.005]

        # Normalize the scores and make human friendly
        df.score /= np.abs(df.score).sum()
        df.score *= 100
        
        class_names = LE.classes_
        df["negative_class"] = class_names[0]
        df["positive_class"] = class_names[1]

        data.append(df)
        
    df = pd.concat(data).set_index(["negative_class","positive_class"])
    df.to_csv(f_csv_out)
