import predictions as pred
import numpy as np
import pandas as pd
import h5py
import os
import itertools
import collections
from utils.os_utils import grab_files

from predictions import categorical_predict

ERROR_MATRIX = {}
PREDICTIONS = {}

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("predict")
    score_config = simple_config.load("score")
    import_config = simple_config.load("import_data")
    use_meta = config['use_meta']
    use_reduced = config['use_reduced']

    # For now, we can only deal with one column using meta!
    assert(len(config["categorical_columns"]) == 1)

    f_h5 = os.path.join(
        score_config["output_data_directory"],
        score_config["document_scores"]["f_db"],
    )

    h5 = h5py.File(f_h5, 'r')

    methods = h5.keys()
    pred_dir = import_config["output_data_directory"]
    pred_files = grab_files('*.csv', pred_dir)
    pred_col = config["target_column"]

    # Load the categorical columns
    cols = ['_ref', ] + config["categorical_columns"]
    ITR = (pd.read_csv(x, usecols=cols).set_index('_ref') for x in pred_files)
    df = pd.concat(list(ITR))

    ITR = itertools.product(methods, config["categorical_columns"])

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
        _ref = g["_ref"][:]

        Y = np.hstack(df[cat_col].values)
        counts = np.array(collections.Counter(Y).values(), dtype=float)
        counts /= counts.sum()
        print("  Class balance for catergorical prediction: ", counts)

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores, F1, errors, pred = categorical_predict(X, Y, method, config)

        text = "  F1 {:0.3f}; Accuracy {:0.3f}; baseline ({:0.3f})"
        print(text.format(scores.mean(), F1.mean(), baseline_score))

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

    if use_meta:
        # Build meta predictor
        META_X = np.hstack([PREDICTIONS[method] for method
                            in config["meta_methods"]])

        method = "meta"

        text = "Predicting [{}] [{}:{}]"
        print(text.format(method, cat_col, pred_col))

        scores, F1, errors, pred = categorical_predict(META_X, Y,
                                                       method, config)

        text = "  F1 {:0.3f}; Accuracy {:0.3f}; baseline ({:0.3f})"
        print(text.format(scores.mean(), F1.mean(), baseline_score))

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors


names = methods

if use_meta:
    names += ["meta", ]

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

import seaborn as sns
plt = sns.plt
sns.heatmap(df, annot=True, vmin=0, vmax=1.2 * max_offdiagonal, fmt="d")
plt.show()
