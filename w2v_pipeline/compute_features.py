from gensim.models.word2vec import Word2Vec
from utils.os_utils import grab_files, mkdir
from utils.db_utils import database_iterator

import sqlite3, os, itertools
import numpy as np

input_table = "parsed5_pos_tokenized"

_DEFAULT_IMPORT_DIRECTORY = "sql_data"
_DEFAULT_EXPORT_DIRECTORY = "models"

target_columns = ["abstract", "specificAims"]

limit_global = 0
epoch_n = 20

#MODEL_SIZES = [100,200,300,400]
#MIN_COUNTS  = [30,20,10,5]

MODEL_SIZES = [200,]
MIN_COUNTS  = [30,]

F_SQL = grab_files("*.sqlite", _DEFAULT_IMPORT_DIRECTORY)

######################################################################

def corpus_ITR(F_SQL):

    # Iterate over the corpus, taking a random path each time

    randomized_fileset = np.random.permutation(len(F_SQL))
    F_SQL2 = [F_SQL[i] for i in randomized_fileset]

    # Create a list of files and columns
    INPUT_ITR = list(itertools.product(F_SQL, target_columns))

    # Randomize the order
    INPUT_ITR  = [INPUT_ITR[n] for n in 
                  np.random.permutation(len(INPUT_ITR))]

    for item in INPUT_ITR:
        f_sqlite, column = item

        conn = sqlite3.connect(f_sqlite, check_same_thread=False)

        ITR  = database_iterator(column, 
                                 input_table, 
                                 conn, 
                                 limit=limit_global)

        token_list = [x[1].split() for x in ITR]
        k = len(token_list)
        
        for i in np.random.permutation(k):
            yield token_list[i]
        
        conn.close()   

def build_features(model_size, min_count):

    features = Word2Vec(workers=8,
                        window=5,
                        negative=5,
                        sample=1e-5,
                        size=model_size,
                        min_count=min_count)

    print "Learning the vocabulary"
    ITR = corpus_ITR(F_SQL)
    features.build_vocab(ITR)

    print features

    print "Training the features"
    for n in range(epoch_n):
        print " - Epoch {}".format(n)
        ITR = corpus_ITR(F_SQL)
        features.train(ITR)

    print "Reducing the features"
    features.init_sims(replace=True)

    return features


######################################################################

if __name__ == "__main__":

    mkdir(_DEFAULT_EXPORT_DIRECTORY)

    for ms,mc in itertools.product(MODEL_SIZES,MIN_COUNTS):

        f = "models/size_{}_min_{}.word2vec"
        f_features = f.format(ms,mc)

        if os.path.exists(f_features):
            print "Skipping", f_features
            continue

        print "Building", f_features

        features = build_features(ms,mc)

        print "Saving the features"
        features.save(f_features)
