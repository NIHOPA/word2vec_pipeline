import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
import sklearn.ensemble 

def categorical_predict(X,y_org,config):

    # Make sure the sizes match
    msg = "X shape {}, y_org shape {} (mismatch!)"
    assert X.shape[0] == y_org.shape[0], msg.format(X.shape[0],
                                                    y_org.shape[0])

    enc = LabelEncoder()
    y = enc.fit_transform(y_org)

    print " Number of unique entries in [y]", np.unique(y).shape[0]
    
    clf_args = {
        "n_jobs" : -1,
        "n_estimators" : int(config["n_estimators"]),
    }
    
    clf = sklearn.ensemble.ExtraTreesClassifier(**clf_args)

    CV_args = {
        "cv" : int(config["cross_validation_folds"]),
        "n_jobs" : 10,
    }

    scores = cross_val_score(clf,X,y, **CV_args)

    return scores

    
    

    exit()
    clf.fit(X,y)

    
    
    exit()
