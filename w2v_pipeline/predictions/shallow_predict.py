import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import sklearn.ensemble

from utils.parallel_utils import jobmap

def clf_extratree_predictor(item):
    (clf_args,idx,X,y) = item
    train_index, test_index = idx

    clf = sklearn.ensemble.ExtraTreesClassifier(**clf_args)
        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train,y_train)
    
    pred   = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)

    return idx,pred,pred_proba


def categorical_predict(X,y_org,method_name,config):

    # Make sure the sizes match
    msg = "X shape {}, y_org shape {} (mismatch!)"
    assert X.shape[0] == y_org.shape[0], msg.format(X.shape[0],
                                                    y_org.shape[0])

    enc = LabelEncoder()
    y = enc.fit_transform(y_org)

    label_n = np.unique(y).shape[0]
    msg = "[{}] number of unique entries in [y {}]: {}"
    print msg.format(method_name, X.shape, label_n)
    
    clf_args = {
        "n_jobs" : -1,
        "n_estimators" : int(config["n_estimators"]),
    }
    

    skf = StratifiedKFold(y,
                          n_folds=10,
                          shuffle=False)
    scores = []

    INPUT_ITR = ((clf_args, idx, X, y) for idx in skf)

    ITR = jobmap(clf_extratree_predictor, INPUT_ITR, True)

    error_counts   = np.zeros(y.size,dtype=float)
    predict_scores = np.zeros([y.size,label_n],dtype=float)

    for result in ITR:
        idx,pred,pred_proba = result
        train_index, test_index = idx

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        errors = y_test!=pred
        
        scores.append(1-errors.mean())        
        error_counts[test_index[errors]] += 1

        predict_scores[test_index] = pred_proba

    # For StratifiedKFold, each test set is hit only once
    # so normalization is simple
    error_counts /= 1.0

    return np.array(scores), error_counts, predict_scores
