import numpy as np
import itertools, multiprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import sklearn.ensemble

def clf_extratree_predictor(item):
    (clf_args,idx,X,y) = item
    train_index, test_index = idx

    clf = sklearn.ensemble.ExtraTreesClassifier(**clf_args)
        
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train,y_train)
    
    pred   = clf.predict(X_test)

    return clf,idx,pred


def categorical_predict(X,y_org,method_name,config):

    # Make sure the sizes match
    msg = "X shape {}, y_org shape {} (mismatch!)"
    assert X.shape[0] == y_org.shape[0], msg.format(X.shape[0],
                                                    y_org.shape[0])

    enc = LabelEncoder()
    y = enc.fit_transform(y_org)

    msg = "[{}] number of unique entries in [y {}]: {}"
    print msg.format(method_name, X.shape, np.unique(y).shape[0])
    
    clf_args = {
        "n_jobs" : -1,
        "n_estimators" : int(config["n_estimators"]),
    }
    

    skf = StratifiedKFold(y,
                          n_folds=10,
                          shuffle=False)
    scores = []

    INPUT_ITR = ((clf_args, idx, X, y) for idx in skf)

    ITR = itertools.imap(clf_extratree_predictor, INPUT_ITR)
    MP = multiprocessing.Pool()
    ITR = MP.imap(clf_extratree_predictor, INPUT_ITR)

    for result in ITR:
        clf,idx,pred = result
        train_index, test_index = idx

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        errors = y_test==pred
        scores.append(errors.mean())

    return np.array(scores)

    '''
    CV_args = {
        "cv" : int(config["cross_validation_folds"]),
        "n_jobs" : 10,
    }
    scores = cross_val_score(clf,X,y,**CV_args)
    '''
