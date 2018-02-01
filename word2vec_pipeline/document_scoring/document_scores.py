import collections
import itertools
import os
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.data_utils import load_w2vec, touch_h5, load_document_vectors

def L2_norm(doc_vec):
    # Renormalize onto the hypersphere
    doc_vec /= np.linalg.norm(doc_vec)

    # Sanity check, L2 norm and set to zeros if not
    if doc_vec.any():
        assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
    else:
        print("Warning L2 norm not satisifed (zero-vector returned)")
        doc_vec = np.zeros(doc_vec.shape)

    return doc_vec

def token_counts(tokens, size_mb=1):
    '''
    Returns a count for the number of times a token appears in a list.
    bounter is slower here since we aren't counting a large enough corpus.
    '''
    return collections.Counter(tokens)

#####################################################################################
    
class generic_document_score(object):

    def __init__(self,
                 negative_weights=None,
                 *args, **kwargs):

        # Load the model from disk
        self.M = load_w2vec()

        # Build the dictionary, and a mapping from word2index
        self.shape = self.M.wv.syn0.shape
        self.vocab = dict(zip(self.M.wv.index2word, xrange(self.shape[0])))

        if negative_weights:
            
            NV = []
            for word,weight in negative_weights.items():

                if not self.check_word_vector(word):
                    msg = "Negative weight word '{}' not found in dictionary"
                    print(msg.format(word))
                    continue
                
                vec = self.get_word_vector(word)
                scale = np.exp(-float(weight) * self.M.wv.syn0.dot(vec))

                # Don't oversample, max out weights to unity
                scale[scale > 1] = 1.0
                NV.append(scale)
            self.negative_weights = np.array(NV).T.sum(axis=1)
            
        else:
            self.negative_weights = np.ones(len(self.vocab), dtype=float)

        # Make sure nothing has been set yet
        self.V = self._ref = None
        self.h5py_args = {"compression":"gzip"}
        
    def _empty_vector(self):
        return np.zeros((self.shape[1],), dtype=float)
    
    def check_word_vector(self, word):
        # Reuturns True/False if the word vector is in the vocab
        return word in self.vocab

    def get_word_vector(self, word):
        return self.M[word].astype(np.float64)
    
    def get_negative_word_weight(self, word):
        return self.negative_weights[self.vocab[word]]
    
    def get_word_vectors(self, ws):
        return np.array([self.get_word_vector(w) for w in ws])

    def get_negative_word_weights(self, ws):
        return np.array([self.get_negative_word_weight(w) for w in ws])

    def get_counts(self, ws):
        return np.array([ws[w] for w in ws])

    def get_tokens_from_text(self, text):
        tokens = text.split()
        
        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.vocab]

        if not valid_tokens:
            raise Warning("No valid tokens in document!")
        
        return valid_tokens

    def __call__(self, text):
        raise NotImplementedError

    def get_h5save_object(self, f_db):
        # Returns a usable h5 object to store data
        h5 = touch_h5(f_db)
        g  = h5.require_group(self.method)
        return g
    
    def save_h5(self, h5, col, data):
        # Saves (or overwrites) a column in an h5 object
        if col in h5:
            del h5[col]        
        return h5.create_dataset(col, data=data, **self.h5py_args)

    def save(
            self,
            data,
            f_csv,
            f_db,
    ):
        '''
        Takes in a dictionary of _ref:doc_vec and saves it to an h5 file.
        Save only to the basename of the file.
        '''
        
        _refs = sorted(data.keys())
        V = np.array([data[r] for r in _refs])

        # Sanity check, should not have any NaNs
        assert(not np.isnan(V).any())

        # Set the size explictly as a sanity check
        size_n, dim_V = V.shape

        g = self.get_h5save_object(f_db)
        gx = g.require_group(os.path.basename(f_csv))

        self.save_h5(gx, "V", V)
        self.save_h5(gx, "_ref", _refs)

    def compute_reduced_representation(self, f_db, n_components=10):

        # Only load the library if we are performing PCA
        from sklearn.decomposition import IncrementalPCA

        DV = load_document_vectors(self.method)
        V = DV["docv"]
        clf = IncrementalPCA(n_components=n_components)

        msg = "Performing PCA on {}, ({})->({})"
        print(msg.format(self.method, V.shape[1], n_components))
        VX = clf.fit_transform(V)

        g = self.get_h5save_object(f_db)
        for key in g.keys():
            idx = g[key]["_ref"][:]
            
            self.save_h5(g[key], "VX", VX[idx, :])
            self.save_h5(g[key], "VX_explained_variance_ratio_", clf.explained_variance_ratio_)
            self.save_h5(g[key], "VX_components_", clf.components_)


#####################################################################################


class IDF_document_score(generic_document_score):
    
    def __init__(self,
                 output_data_directory=None,
                 term_document_frequency=None,
                 *args, **kwargs):
        
        super(IDF_document_score, self).__init__(
            output_data_directory=output_data_directory, *args, **kwargs)

        assert(term_document_frequency is not None)

        f_csv = os.path.join(
            output_data_directory,
            term_document_frequency["f_db"],
        )
        
        IDF = pd.read_csv(f_csv)
        IDF = dict(zip(IDF["word"].values, IDF["count"].values))
        self.corpus_N = IDF.pop("__pipeline_document_counter")

        # Compute the IDF
        for key in IDF:
            IDF[key] = np.log(float(self.corpus_N) / (IDF[key] + 1))
        self.IDF = IDF

    def get_IDF_weight(self, w):
        if w in self.IDF:
            return self.IDF[w]
        else:
            return 0.0

    def get_IDF_weights(self, ws):
        return np.array([self.get_IDF_weight(w) for w in ws])

###############################################################################


class score_simple(generic_document_score):
    method = "simple"

    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        counts = token_counts(tokens)

        W = self.get_word_vectors(counts)
        n = self.get_negative_word_weights(counts)
        C = self.get_counts(counts)
        
        return L2_norm(((C*n)*W.T).sum(axis=1))

class score_unique(generic_document_score):
    method = 'unique'

    def __call__(self, text):
        tokens = set(self.get_tokens_from_text(text))

        W = self.get_word_vectors(tokens)
        n = self.get_negative_word_weights(tokens)
        
        return L2_norm((n*W.T).sum(axis=1))

class score_simple_IDF(IDF_document_score):
    method = "simple_IDF"
    
    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        counts = token_counts(tokens)
        
        W = self.get_word_vectors(counts)
        n = self.get_negative_word_weights(counts)
        C = self.get_counts(counts)
        I = self.get_IDF_weights(counts)

        return L2_norm(((I*C*n)*W.T).sum(axis=1))

    

class score_unique_IDF(IDF_document_score):
    method = "unique_IDF"

    def __call__(self, text):
        tokens = set(self.get_tokens_from_text(text))
        
        W = self.get_word_vectors(tokens)
        n = self.get_negative_word_weights(tokens)
        I = self.get_IDF_weights(tokens)

        return L2_norm(((I*n)*W.T).sum(axis=1))
