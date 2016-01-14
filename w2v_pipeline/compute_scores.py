from gensim.models.word2vec import Word2Vec
import h5py
import numpy as np
import scipy.special

import os, itertools, collections, argparse, sqlite3
from utils.os_utils import grab_files, mkdir
from utils.db_utils import database_iterator

parser = argparse.ArgumentParser()
parser.add_argument("-f","--force",default=False,action="store_true")
parser.add_argument("-d","--debug",default=False,action="store_true")
parser.add_argument("-p","--parallel",default=False,action="store_true")
args = parser.parse_args()

input_table = "parsed5_pos_tokenized"
_DEFAULT_MODEL_DIRECTORY  = "models"
_DEFAULT_SQL_DIRECTORY    = "sql_data"
_DEFAULT_EXPORT_DIRECTORY = "collated"
f_output = "scores.h5"

w2v_summations = ["simple", "unique", "TF_IDF"]

## Using both config parser and argparse, should fix this
from utils.config_reader import load_config
cargs = load_config()
target_columns = cargs["target_columns"]
_DEBUG = cargs["debug"]

F_MODELS = grab_files("*.word2vec",_DEFAULT_MODEL_DIRECTORY)
limit_global = 0
global_debug = False + args.debug
global_parallel = args.parallel

f_TF_db = "collated/TF.sqlite"

# Load the TF global model 
conn_TF = sqlite3.connect(f_TF_db,check_same_thread=False)
cmd = "SELECT word, count FROM DF"
cursor = conn_TF.execute(cmd)
document_freq = dict(cursor)
total_documents = document_freq[""]
IDF = {}
for word,count in document_freq.items():
    IDF[word] = np.log(total_documents / float(count))
del document_freq

M = None
if global_debug: limit_global = 100

######################################################################


def von_Mises_Fisher_kappa(X):
    # points,dimension
    n,p = map(float,X.shape)

    # Estimates kappa from Banerjee (crude approximation)    
    R = np.linalg.norm(X.sum(axis=0)/n)
    if R==1: return 10**4
    k0 = (R*(p-R**2))/(1-R**2)
    return k0

    # Estimates from Survit Sra (not used, error is only ~ 0.2)
    def AP(k):
        num = scipy.special.iv(p/2,k)
        dem = scipy.special.iv(p/2-1,k)
        try:
            val = num/dem
        except RuntimeWarning:
            val = 1.0
        return val

    if k0>10**3: return k0

    apk0 = AP(k0)
    k1 = k0 - (apk0-R) / (1-apk0**2 - ((p-1)/k0)*apk0)

    apk1 = AP(k1)
    k2 = k1 - (apk1-R) / (1-apk1**2 - ((p-1)/k1)*apk1)
    
    if np.isnan(k2): return k0
    return k2

class word2vec_score_model(object):
    def __init__(self, f_model, method=None):

        # Load the model from disk
        self.M = Word2Vec.load(f_model)       
        self.shape = self.M.syn0.shape

        # Build total counts
        self.counts = {}
        for key,val in self.M.vocab.items():
            self.counts[key] = val.count

        if method is not None:
            self.set_w2v_method(method)

    def set_w2v_method(self, method):
        self.w2v_method = method

    def score_document(self,tokens):

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.M]
        local_counts = collections.Counter(valid_tokens)
        tokens = set(valid_tokens)


        # If doc_vec is empty choose random vector
        if not len(tokens):
            doc_vec = np.random.uniform(-1,1,size=self.shape[1])
            doc_vec /= np.linalg.norm(doc_vec)

            # Add on one extra dimension
            kappa = np.random.uniform(0,10**4)
            extra_info = [kappa,]
            return doc_vec, extra_info
        
        local_total = float(sum(local_counts.values()))

        method = self.w2v_method

        if method in ["unique"]:
            weights = dict.fromkeys(tokens, 1.0)
        elif method in ["simple"]:
            weights = dict([(w,local_counts[w]) for w in tokens])
        elif method in ["TF_IDF"]:
            weights = dict([(w,IDF[w]*c) 
                            for w,c in local_counts.items()])
        else:
            print "UNKNOWN w2v method", method         

        # Lookup each word that is in the model
        DV = np.array([self.M[w] for w in tokens])
        W  = np.array([weights[w] for w in tokens]).reshape(-1,1)

        # Sum all the vectors with their weights
        doc_vec = (W*DV).sum(axis=0)        
        
        # Renormalize onto the hypersphere
        doc_vec /= np.linalg.norm(doc_vec)

        # Sanity check
        assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
       
        # Add on extra dimensions
        kappa = von_Mises_Fisher_kappa(DV)
        extra_info = [kappa,]
        return doc_vec, extra_info

    def get_vocabulary(self):
        return self.M.index2word

    def get_embedding_matrix(self):
        return self.M.syn0

    def get_counts_vector(self):
        return [self.counts[x] for x in self.get_vocabulary()]

    def __call__(self, tokens):
        return self.score_document(tokens)

    def score_iterator(self, ITR):
        # Scores the second item of the iterator
        return np.array([self(item[1]) for item in ITR])

#####################################################################

class scorer(object):
    def __init__(self, M):
        self.M = M
    
    def __call__(self, document_iterator):
        SCORES, EXTRA_INFO = [], []
        
        for idx, tokens in document_iterator:
            score, extra_info = self.M(tokens)
            SCORES.append(score)
            EXTRA_INFO.append(extra_info)

        return SCORES, EXTRA_INFO
    

def token_iterator(item):
    f_sqlite, column = item
    conn = sqlite3.connect(f_sqlite, check_same_thread=False)
    ITR  = database_iterator(column, 
                             input_table, 
                             conn, 
                             limit=limit_global)
    for idx, text in ITR:
        yield idx, text.split()

if global_parallel:
    import multiprocessing
    MP = multiprocessing.Pool()


#########################################################################

if __name__ == "__main__":

    # Check if output exists and we are not forcing computation
    f_h5 = os.path.join(_DEFAULT_EXPORT_DIRECTORY, f_output)
    mkdir(_DEFAULT_EXPORT_DIRECTORY)

    if os.path.exists(f_h5) and not args.force:
        print "File {} already exists, exiting".format(f_h5)
        exit(1)
        
    h5 = h5py.File(f_h5,'w')


    for f_model in F_MODELS:

        print "Starting model", f_model

        M = word2vec_score_model(f_model)
        g = h5.create_group(f_model)

        S = scorer(M)

        # Save the embedding matrix
        g.create_dataset("syn0", data=M.get_embedding_matrix(),
                         compression='gzip')

        # Save the vocabulary
        vocab = np.array(M.get_vocabulary(),dtype="S40")
        g.create_dataset("vocab",data=vocab)

        # Save the counts
        g.create_dataset("counts",data=M.get_counts_vector())
        g2 = g.create_group("embeddings")

        for w2v_method in w2v_summations:

            print "Starting method", w2v_method

            M.set_w2v_method(w2v_method)

            g3 = g2.create_group(w2v_method)

            F_SQL = grab_files("*.sqlite", _DEFAULT_SQL_DIRECTORY)

            INPUT_ITR  = list(itertools.product(F_SQL, target_columns))
            TOKEN_ITRS = [token_iterator(item) for item in INPUT_ITR]
            RESULT_ITR = itertools.imap(S, TOKEN_ITRS)

            if global_parallel:
                RESULT_ITR = MP.imap(S, TOKEN_ITRS)

            for item,result in zip(INPUT_ITR, RESULT_ITR):

                SCORES, EXTRA_INFO = result

                f_sqlite, col = item
                group_name = os.path.basename(f_sqlite) + '/' + col
                g4 = g3.create_group(group_name)           

                g4.create_dataset("scores", data=SCORES, 
                                  compression="gzip")
                g4.create_dataset("extra_info", data=EXTRA_INFO, 
                                  compression="gzip")

                print "Completed", item
