import predictions as pred
import numpy as np
import pandas as pd
import h5py
import os, glob, itertools, collections
from sqlalchemy import create_engine

from predictions import categorical_predict

ERROR_MATRIX = {}
PREDICTIONS = {}

if __name__ == "__main__":

    import simple_config
    config = simple_config.load("predict")
    score_config = simple_config.load("score")
    import_config = simple_config.load("import_data")

    # For now, we can only deal with one column using meta!
    assert(len(config["categorical_columns"])==1)

    f_h5 = os.path.join(
        score_config["output_data_directory"],
        score_config["document_scores"]["f_db"],
    )
    
    h5 = h5py.File(f_h5,'r')

    methods = h5.keys()
    pred_dir = import_config["output_data_directory"]

    input_glob  = os.path.join(pred_dir,'*')
    input_files = glob.glob(input_glob)
    input_names = ['.'.join(os.path.basename(x).split('.')[:-1])
                   for x in input_files]


    ITR = itertools.product(methods,
                            config["categorical_columns"],
                            config["target_columns"])

    for (method, cat_col, data_col) in ITR:

        assert(method in h5)    
        assert(data_col in h5[method])
        g = h5[method][data_col]
    
        # Make sure every file has been scored or skip it.
        saved_input_names = []
        for f in input_names:
            if f not in g:
                msg = "'{}' not in {}:{} skipping"
                print msg.format(f,method,cat_col)
                continue
            saved_input_names.append(f)


        # Load document score data
        X = np.vstack([g[name][:]
                       for name in saved_input_names])

        # Load the categorical columns
        Y = []
        for name in saved_input_names:
            f_sql = os.path.join(pred_dir,name) + '.sqlite'
            engine = create_engine('sqlite:///'+f_sql)
            df = pd.read_sql_table("original",engine,
                                   columns=[cat_col,])
            y = df[cat_col].values
            Y.append(y)

        Y = np.hstack(Y)

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores,errors,pred = categorical_predict(X,Y,method,config)

        text = "Predicting [{}] [{}:{}] {:0.4f} ({:0.4f})"
        print text.format(method, cat_col, data_col,
                          scores.mean(), baseline_score)

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

    # Build meta predictor
    META_X = np.hstack([PREDICTIONS[method] for method
                        in config["meta_methods"]])
    
    method = "meta"
    scores,errors,pred = categorical_predict(META_X,Y,method,config)

    text = "Predicting [{}] [{}:{}] {:0.4f} ({:0.4f})"
    print text.format(method, cat_col, data_col,
                      scores.mean(), baseline_score)

    PREDICTIONS[method] = pred
    ERROR_MATRIX[method] = errors


names = methods + ["meta",]
df = pd.DataFrame(0, index=names,columns=names)

max_offdiagonal = 0
for na,nb in itertools.product(names,repeat=2):
    if na!=nb:
        idx = (ERROR_MATRIX[na]==0) * (ERROR_MATRIX[nb]==1)
        max_offdiagonal = max(max_offdiagonal, idx.sum())
    else:
        idx = ERROR_MATRIX[na]==0

    df[na][nb] = idx.sum()

print df

import seaborn as sns
plt = sns.plt
sns.heatmap(df,annot=True,vmin=0,vmax=1.2*max_offdiagonal,fmt="d")
plt.show()
