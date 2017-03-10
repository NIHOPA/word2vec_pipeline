import numpy as np
import pandas as pd
import h5py
import os
import itertools
import collections

from utils.os_utils import grab_files, mkdir
from utils.data_utils import load_ORG_data
from predictions import categorical_predict

import seaborn as sns

def predict_from_config(config):

    ERROR_MATRIX = {}
    PREDICTIONS = {}

    use_meta = config["predict"]['use_meta']
    use_reduced = config["predict"]['use_reduced']

    # For now, we can only deal with one column using meta!
    assert(len(config["predict"]["categorical_columns"]) == 1)

    f_h5 = os.path.join(
        config["score"]["output_data_directory"],
        config["score"]["document_scores"]["f_db"],
    )

    h5 = h5py.File(f_h5, 'r')

    methods = h5.keys()
    pred_dir = config["import_data"]["output_data_directory"]
    pred_files = grab_files('*.csv', pred_dir)
    pred_col = config["target_column"]

    pred_output_dir = config["predict"]["output_data_directory"]
    extra_cols = config["predict"]["extra_columns"]
    mkdir(pred_output_dir)

    # Load the categorical columns
    cols = ['_ref', ] + config["predict"]["categorical_columns"]
    ITR = (pd.read_csv(x, usecols=cols).set_index('_ref') for x in pred_files)
    df = pd.concat(list(ITR))

    ITR = itertools.product(methods, config["predict"]["categorical_columns"])

    X_META = []

    cfg = config["predict"]
    cfg["_PARALLEL"] = config["_PARALLEL"]

    df_scores = None

    for (method, cat_col) in ITR:

        text = "Predicting [{}] [{}:{}]"
        print(text.format(method, cat_col, pred_col))

        assert(method in h5)
        g = h5[method]

        # Load document score data
        if use_reduced:
            X = g["VX"][:]
        else:
            X = g["V"][:]

        if use_meta:
            X_META.append(X)

        _ref = g["_ref"][:]
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

    plt = sns.plt
    sns.heatmap(df, annot=True, vmin=0, vmax=1.2 * max_offdiagonal, fmt="d")
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    plt.show()

if __name__ == "__main__":

    import simple_config
    config = simple_config.load()
    predict_from_config(config)

