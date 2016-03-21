import clustering
import numpy as np
import pandas as pd
import h5py
import os, glob, itertools, collections
from sqlalchemy import create_engine

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("clustering")

    print config
    exit()

    '''
    # For now, we can only deal with one column using meta!
    assert(len(config["categorical_columns"])==1)

    f_h5 = config["f_db_scores"]
    h5 = h5py.File(f_h5,'r')

    methods = h5.keys()
    pred_dir = config["predict_target_directory"]

    input_glob  = os.path.join(pred_dir,'*')
    input_files = glob.glob(input_glob)
    input_names = ['.'.join(os.path.basename(x).split('.')[:-1])
                   for x in input_files]

    ITR = itertools.product(methods, config["categorical_columns"])

    for (method, column) in ITR:

        # Make sure every file has been scored or skip it.
        saved_input_names = []
        for f in input_names:
            if f not in h5[method]:
                msg = "'{}' not in {}:{} skipping"
                print msg.format(f,method,column)
                continue
            saved_input_names.append(f)

        # Load document score data
        X = np.vstack([h5[method][name][:]
                       for name in saved_input_names])

        # Load the categorical columns
        Y = []
        for name in saved_input_names:
            f_sql = os.path.join(pred_dir,name) + '.sqlite'
            engine = create_engine('sqlite:///'+f_sql)
            df = pd.read_sql_table("original",engine,
                                   columns=[column,])
            y = df[column].values
            Y.append(y)

        Y = np.hstack(Y)

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores,errors,pred = categorical_predict(X,Y,method,config)

        text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
        print text.format(method, column,
                          scores.mean(), baseline_score)

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

    # Build meta predictor
    META_X = np.hstack([PREDICTIONS[method] for method
                        in config["meta_methods"]])
    
    method = "meta"
    scores,errors,pred = categorical_predict(META_X,Y,method,config)

    text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
    print text.format(method, column,
                      scores.mean(), baseline_score)

    PREDICTIONS[method] = pred
    ERROR_MATRIX[method] = errors
    '''

