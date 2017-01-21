import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import sklearn.ensemble

from utils.parallel_utils import jobmap
from imblearn.over_sampling import SMOTE


def clf_extratree_predictor(item):
    (clf_args, idx, X, y, use_SMOTE) = item
    train_index, test_index = idx

    clf = sklearn.ensemble.ExtraTreesClassifier(**clf_args)

    X_train, X_test = X[train_index], X[test_index]
    y_train = y[train_index]

    if use_SMOTE:
        sampler = SMOTE(ratio='auto', kind='regular')
        X_train, y_train = sampler.fit_sample(X_train, y_train)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)

    return idx, pred, pred_proba


def categorical_predict(X, y_org, method_name, config):

    # Make sure the sizes match
    msg = "X shape {}, y_org shape {} (mismatch!)"
    assert X.shape[0] == y_org.shape[0], msg.format(X.shape[0],
                                                    y_org.shape[0])

    enc = LabelEncoder()
    y = enc.fit_transform(y_org)

    label_n = np.unique(y).shape[0]
    # msg = "[{}] number of unique entries in [y {}]: {}"
    # print msg.format(method_name, X.shape, label_n)

    use_SMOTE = config["use_SMOTE"]
    if use_SMOTE:
        print("  Adjusting class balance using SMOTE")

    is_PARALLEL = config["_PARALLEL"]

    clf_args = {
        "n_jobs": -1 if is_PARALLEL else 1,
        "n_estimators": int(config["n_estimators"]),
    }

    skf = StratifiedKFold(n_splits=10, shuffle=False).split(X, y)
    scores = []
    F1_scores = []

    INPUT_ITR = ((clf_args, idx, X, y, use_SMOTE) for idx in skf)

    ITR = jobmap(clf_extratree_predictor, INPUT_ITR, True)

    error_counts = np.zeros(y.size, dtype=float)
    predict_scores = np.zeros([y.size, label_n], dtype=float)

    for result in ITR:
        idx, pred, pred_proba = result
        train_index, test_index = idx

        y_test = y[test_index]

        errors = y_test != pred

        scores.append(1 - errors.mean())
        error_counts[test_index[errors]] += 1

        F1_scores.append(f1_score(y_test, pred))
        predict_scores[test_index] = pred_proba

    # For StratifiedKFold, each test set is hit only once
    # so normalization is simple
    error_counts /= 1.0

    return np.array(scores), np.array(F1_scores), error_counts, predict_scores
