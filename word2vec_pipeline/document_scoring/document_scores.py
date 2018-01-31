import collections
import itertools
import os
import joblib
import simple_config
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA
from utils.data_utils import load_w2vec, touch_h5

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

    

class generic_document_score(object):

    def __init__(self, *args, **kwargs):

        # Load the model from disk
        self.M = load_w2vec()

        # Build the dictionary, and a mapping from word2index
        self.shape = self.M.wv.syn0.shape
        self.vocab = dict(zip(self.M.wv.index2word, xrange(self.shape[0])))

        if "negative_weights" in kwargs:
            NV = []
            for word,weight in kwargs["negative_weights"].items():

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
            self.negative_weights = np.ones(vocab_n, dtype=float)

        # Save the target column to compute
        self.target_column = simple_config.load()["target_column"]

        # Make sure nothing has been set yet
        self.V = self._ref = None

        # Set the variables for reduced representation
        config_score = simple_config.load()["score"]
        self.compute_reduced = config_score["compute_reduced_representation"]

        if self.compute_reduced:
            sec = config_score['reduced_representation']
            self.reduced_n_components = sec['n_components']

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
    
    '''
    def _compute_item_weights(self, **da):
        msg = "UNKNOWN w2v weights {}".format(self.method)
        raise KeyError(msg)

    def _compute_embedding_vector(self, **da):
        msg = "UNKNOWN w2v embedding {}".format(self.method)
        raise KeyError(msg)

    def _compute_doc_vector(self, **da):
        msg = "UNKNOWN w2v doc vec {}".format(self.method)
        raise KeyError(msg)
    '''
    
    '''
    def score_document(self, row):
        text = row[self.target_column]
        text = unicode(text)
        tokens = text.split()

        # Document args
        da = {}

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.vocab]

        da["local_counts"] = token_counts(valid_tokens)

        if not da["local_counts"].cardinality():
            msg = "Document (_ref={}, len(text)={}) has no valid tokens!"
            print msg
            #print(msg.format(row["_ref"], len(text)))
            # raise ValueError(msg)
            da["local_counts"] = {}
            
        da["weights"] = self._compute_item_weights(**da)
        da['DV'] = self._compute_embedding_vector(**da)
        da['doc_vec'] = self._compute_doc_vector(**da)

        # Sanity check, should not have any NaNs
        assert(not np.isnan(da['doc_vec']).any())

        row['doc_vec'] = da['doc_vec']

        return row
    
    def compute_single(self, INPUT_ITR):

        assert(self.method is not None)
        print("Scoring {}".format(self.method))

        self._ref = []
        self.V = []
        self.current_filename = None
        ITR = itertools.imap(self.score_document, tqdm(INPUT_ITR))

        for row in ITR:

            # Require that filenames don't change in compute_single
            assert (self.current_filename in [None, row["_filename"]])
            self.current_filename = row["_filename"]

            self.V.append(row["doc_vec"])
            self._ref.append(int(row["_ref"]))

        self.V = np.array(self.V)
        self._ref = np.array(self._ref)


    def save_single(self):

        assert(self.V is not None)
        assert(self._ref is not None)

        # Set the size explictly as a sanity check
        size_n, dim_V = self.V.shape

        config_score = simple_config.load()["score"]
        f_db = os.path.join(
            config_score["output_data_directory"],
            config_score["document_scores"]["f_db"]
        )

        h5 = touch_h5(f_db)
        g  = h5.require_group(self.method)
        gx = g.require_group(self.current_filename)

        # Save the data array
        msg = "Saving {} {} ({})"
        print(msg.format(self.method, self.current_filename, size_n))

        for col in ["V", "_ref", "VX",
                    "VX_explained_variance_ratio_",
                    "VX_components_"]:
            if col in gx:
                #print "  Clearing", self.method, self.current_filename, col
                del gx[col]

        gx.create_dataset("V", data=self.V, **self.h5py_args)
        gx.create_dataset("_ref", data=self._ref, **self.h5py_args)

    def compute_reduced_representation(self):

        if not self.compute_reduced:
            return None

        config_score = simple_config.load()["score"]
        f_db = os.path.join(
            config_score["output_data_directory"],
            config_score["document_scores"]["f_db"]
        )

        h5 = touch_h5(f_db)
        g = h5[self.method]

        keys = g.keys()
        V     = np.vstack([g[x]["V"][:] for x in keys])
        sizes = [g[x]["_ref"].shape[0] for x in keys]
        
        nc = self.reduced_n_components
        clf = IncrementalPCA(n_components=nc)

        msg = "Performing PCA on {}, ({})->({})"
        print(msg.format(self.method, V.shape[1], nc))

        VX = clf.fit_transform(V)
        EVR = clf.explained_variance_ratio_
        COMPONENTS = clf.components_

        for key, size in zip(keys, sizes):

            # Take slices equal to the size
            vx, VX = VX[:size,:], VX[size:, :]
            evr, EVR = EVR[:size], EVR[size:]
            com, COMPONENTS = COMPONENTS[:size,:], COMPONENTS[size:, :]

            g[key].create_dataset("VX", data=vx, **self.h5py_args)
            g[key].create_dataset("VX_explained_variance_ratio_", data=evr)
            g[key].create_dataset("VX_components_", data=com)

        h5.close()
    '''


class IDF_document_score(generic_document_score):
    
    def __init__(self, *args, **kwargs):
        super(IDF_document_score, self).__init__(*args, **kwargs)

        f_db = os.path.join(
            kwargs['output_data_directory'],
            kwargs['term_frequency']['f_db']
        )
        if not os.path.exists(f_db):
            msg = "{} not computed yet, needed for TF methods!"
            raise ValueError(msg.format(f_db))
        
        score_config = simple_config.load()["score"]
        f_csv = os.path.join(
            score_config["output_data_directory"],
            score_config["term_document_frequency"]["f_db"],
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

class score_simple_TF(IDF_document_score):
    method = "simple_TF"
    
    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        counts = token_counts(tokens)
        
        W = self.get_word_vectors(counts)
        n = self.get_negative_word_weights(counts)
        C = self.get_counts(counts)
        I = self.get_IDF_weights(counts)

        return L2_norm(((I*C*n)*W.T).sum(axis=1))

    

class score_unique_TF(score_simple_TF):
    method = "unique_TF"

    def __call__(self, text):
        tokens = set(self.get_tokens_from_text(text))
        
        W = self.get_word_vectors(tokens)
        n = self.get_negative_word_weights(tokens)
        I = self.get_IDF_weights(tokens)

        return L2_norm(((I*n)*W.T).sum(axis=1))


#

###############################################################################

'''
class new_score_unique_TF(score_simple_TF):
    method = "unique_TF"

    def __init__(self):
        
        # Load the model from disk
        self.M = load_w2vec()

        # Find the words known
        self.vocab = set(self.M.wv.index2word)
        self.shape = self.M.wv.syn0.shape

        # Build the dictionary
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.wv.index2word, range(vocab_n)))

        print "HI!"
'''
