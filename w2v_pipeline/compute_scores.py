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
args = parser.parse_args()

input_table = "parsed5_pos_tokenized"
_DEFAULT_MODEL_DIRECTORY  = "models"
_DEFAULT_SQL_DIRECTORY    = "sql_data"
_DEFAULT_EXPORT_DIRECTORY = "collated"
f_output = "scores.h5"

target_columns = ["abstract", "specificAims"]
F_MODELS = grab_files("*.word2vec",_DEFAULT_MODEL_DIRECTORY)

limit_global = 0
global_debug = False + args.debug

#args.force = True
#global_debug = True

######################################################################

global_w2v_summation = "simple"
#global_w2v_summation = "unique"
#global_w2v_summation = "log"
#global_w2v_summation = "local_TF_IDF"
#global_w2v_summation = "TF_IDF"
#global_w2v_summation = "ratio_TF_IDF"
#global_w2v_summation = "square"
#global_w2v_summation = "max_eigen"

if global_debug: limit_global = 100

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
    def __init__(self, f_model):

        # Load the model from disk
        self.M = Word2Vec.load(f_model)       
        self.shape = self.M.syn0.shape

        # Build TF_IDF
        self.counts = {}
        for key,val in self.M.vocab.items():
            self.counts[key] = val.count

        total = float(sum(self.counts.values()))
        self.TF_IDF = {}

        for key in self.counts:
            self.TF_IDF[key] = self.counts[key]/total

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

        if global_w2v_summation in ["unique","max_eigen"]:
            weights = dict.fromkeys(tokens, 1.0)
        elif global_w2v_summation in ["simple",]:
            weights = dict([(w,local_counts[w]) for w in tokens])
        elif global_w2v_summation == "square":
            weights = dict([(w,local_counts[w]**2) for w in tokens])
        elif global_w2v_summation == "log":
            weights = dict([(w,np.log(1+local_counts[w])) for w in tokens])
        elif global_w2v_summation == "local_TF_IDF":
            weights = dict([(w,local_counts[w]/local_total) for w in tokens])
        elif global_w2v_summation == "ratio_TF_IDF":
            weights = dict([(w,self.TF_IDF[w]/(local_counts[w]/local_total)) 
                            for w in tokens])
        elif global_w2v_summation == "TF_IDF":
            weights = dict([(w,self.TF_IDF[w]) for w in tokens])           
        else:
            print "UNKNOWN w2v summation method", global_w2v_summation         

        # Lookup each word that is in the model
        DV = np.array([self.M[w] for w in tokens])
        W  = np.array([weights[w] for w in tokens]).reshape(-1,1)

        if global_w2v_summation == "max_eigen":
            U,s,V = np.linalg.svd(DV)
            doc_vec = np.array([V[0,:]])
            doc_vec = (doc_vec).sum(axis=0)            
        else:
            # Sum all the vectors with their weights
            doc_vec = (W*DV).sum(axis=0)        
        
        # DEBUG FOR DOC2VEC
        #DV = np.array([self.M[w] for w in valid_tokens])
        #doc_vec = self.M.infer_vector(valid_tokens)

        # Renormalize onto the hypersphere
        doc_vec /= np.linalg.norm(doc_vec)

        # Renormalize onto the hypersphere
        #doc_vec /= np.linalg.norm(doc_vec, ord=1)

        # Sanity check
        #assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
       
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

def score_set(M, document_iterator):
    SCORES, EXTRA_INFO = [], []

    for idx, tokens in document_iterator:
        score, extra_info = M(tokens)
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

def compute_model(h5, f_model):
    print "Starting model", f_model

    M = word2vec_score_model(f_model)
    g = h5.create_group(f_model)

    # Save the embedding matrix
    g.create_dataset("syn0", data=M.get_embedding_matrix(),
                     compression='gzip')

    # Save the vocabulary
    vocab = np.array(M.get_vocabulary(),dtype="S40")
    g.create_dataset("vocab",data=vocab)

    # Save the counts
    g.create_dataset("counts",data=M.get_counts_vector())

    g2 = g.create_group("embeddings")

    F_SQL = grab_files("*.sqlite", _DEFAULT_SQL_DIRECTORY)

    INPUT_ITR = itertools.product(F_SQL, target_columns)
    for item in INPUT_ITR:
        f_sqlite, col = item
        print '/'.join(item)
        group_name = os.path.basename(f_sqlite) + '/' + col
        g3 = g2.create_group(group_name)

        document_tokens = token_iterator(item)
        SCORES, EXTRA_INFO  = score_set(M,document_tokens)

        g3.create_dataset("scores", data=SCORES, 
                          compression="gzip")
        g3.create_dataset("extra_info", data=EXTRA_INFO, 
                          compression="gzip")

    #if not global_debug:
    #    ITR = MP.imap(compute_model_file, INPUT_ITR)
    #for (year,scores, extra_info) in ITR:
    #    # Save the scores
    #    g3 = g2.create_group(year)#
    #    g3.create_dataset("scores", data=scores, compression="gzip")
    #    g3.create_dataset("extra_info", data=extra_info, compression="gzip")


#####################################################################

if not global_debug:
    import multiprocessing
    MP = multiprocessing.Pool()

if __name__ == "__main__":

    # Check if output exists and we are not forcing computation
    f_h5 = os.path.join(_DEFAULT_EXPORT_DIRECTORY, f_output)
    mkdir(_DEFAULT_EXPORT_DIRECTORY)

    if os.path.exists(f_h5) and not args.force:
        print "File {} already exists, exiting".format(f_h5)
        exit(1)
    h5 = h5py.File(f_h5,'w')

    for f_model in F_MODELS:
        compute_model(h5, f_model)
