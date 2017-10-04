import numpy as np
import pandas as pd
import os
import itertools
import collections
import pylab as plt

from utils.os_utils import grab_files, mkdir
from utils.data_utils import load_ORG_data, load_document_vectors, get_score_methods
from predictions import categorical_predict

import seaborn as sns


def predict_from_config(config):

    ERROR_MATRIX = {}
    PREDICTIONS = {}

    use_meta = config["predict"]['use_meta']
    use_reduced = config["predict"]['use_reduced']

    # For now, we can only deal with one column using meta!
    assert(len(config["predict"]["categorical_columns"]) == 1)
    
    methods = get_score_methods()
    
    pred_dir = config["import_data"]["output_data_directory"]
    pred_files = grab_files('*.csv', pred_dir)
    pred_col = config["target_column"]

    pred_output_dir = config["predict"]["output_data_directory"]
    extra_cols = config["predict"]["extra_columns"]
    mkdir(pred_output_dir)

    # Load the categorical columns
    df = load_ORG_data(config["predict"]["categorical_columns"])
    ITR = itertools.product(methods, config["predict"]["categorical_columns"])

    X_META = []

    cfg = config["predict"]
    cfg["_PARALLEL"] = config["_PARALLEL"]
    df_scores = None

    for (method, cat_col) in ITR:

        text = "Predicting [{}] [{}:{}]"
        print(text.format(method, cat_col, pred_col))

        DV = load_document_vectors(method)
        X = DV["docv"]
        
        if use_meta:
            X_META.append(X)

        Y = np.hstack(df[cat_col].values)
        counts = np.array(collections.Counter(Y).values(), dtype=float)
        counts /= counts.sum()
        
        # print(" Class balance for catergorical prediction:
        # {}".format(counts))

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores, F1, errors, pred, dfs = categorical_predict(X, Y, method, cfg)

        text = "  F1 {:0.3f}; Accuracy {:0.3f}; baseline ({:0.3f})"
        print(text.format(scores.mean(), F1.mean(), baseline_score))

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

        if df_scores is None:
            df_scores = dfs
        else:
            df_scores[method] = dfs[method]

    if use_meta:
        # Build meta predictor
        # META_X = np.hstack([PREDICTIONS[method] for method
        #                    in config["predict"]["meta_methods"]])
        X_META = np.hstack(X_META)
        method = "meta"

        text = "Predicting [{}] [{}:{}]"
        print(text.format(method, cat_col, pred_col))

        scores, F1, errors, pred, dfs = categorical_predict(X_META, Y,
                                                            method,
                                                            config["predict"])

        text = "  F1 {:0.3f}; Accuracy {:0.3f}; baseline ({:0.3f})"
        print(text.format(scores.mean(), F1.mean(), baseline_score))

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors
        df_scores[method] = dfs[method]

    # Save the predictions
    if extra_cols:
        df_ORG = load_ORG_data(extra_columns=extra_cols)
        for col in extra_cols:
            df_scores[col] = df_ORG[col]

    f_save = os.path.join(pred_output_dir,
                          "{}_prediction.csv".format(cat_col))
    df_scores.index.name = '_ref'
    df_scores.to_csv(f_save)

    names = methods

    if use_meta:
        names += ["meta", ]

    # Plotting methods here

    df = pd.DataFrame(0, index=names, columns=names)

    max_offdiagonal = 0
    for na, nb in itertools.product(names, repeat=2):
        if na != nb:
            idx = (ERROR_MATRIX[na] == 0) * (ERROR_MATRIX[nb] == 1)
            max_offdiagonal = max(max_offdiagonal, idx.sum())
        else:
            idx = ERROR_MATRIX[na] == 0

        df[na][nb] = idx.sum()

    print(df)

    sns.heatmap(df, annot=True, vmin=0, vmax=1.2 * max_offdiagonal, fmt="d")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    plt.show()


if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    predict_from_config(config)
