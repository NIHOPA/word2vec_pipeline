import collections
import itertools
import os
import joblib
import simple_config
import numpy as np
import pandas as pd

from tqdm import tqdm

from locality_hashing import RBP_hasher
from sklearn.decomposition import IncrementalPCA
from utils.mapreduce import corpus_iterator
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

class generic_document_score(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(generic_document_score, self).__init__(*args, **kwargs)

        # Load the model from disk
        self.M = load_w2vec()
        self.shape = self.M.wv.syn0.shape

        # Build the dictionary
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.wv.index2word, range(vocab_n)))

        # Set parallel option (currently does nothing)
        # self._PARALLEL = kwargs["_PARALLEL"]

        if "negative_weights" in kwargs:
            NV = []
            for word,weight in kwargs["negative_weights"].items():
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

    def _compute_item_weights(self, **da):
        msg = "UNKNOWN w2v weights {}".format(self.method)
        raise KeyError(msg)

    def _compute_embedding_vector(self, **da):
        msg = "UNKNOWN w2v embedding {}".format(self.method)
        raise KeyError(msg)

    def _compute_doc_vector(self, **da):
        msg = "UNKNOWN w2v doc vec {}".format(self.method)
        raise KeyError(msg)

    def score_document(self, row):

        text = row[self.target_column]
        text = unicode(text)
        tokens = text.split()

        # Document args
        da = {}

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.M]

        da["local_counts"] = collections.Counter(valid_tokens)
        da["tokens"] = list(set(valid_tokens))


        if not da["tokens"]:
            msg = "Document (_ref={}, len(text)={}) has no valid tokens!"
            print(msg.format(row["_ref"], len(text)))
            # raise ValueError(msg)

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

    def get_word_vector(self, word):
        return self.M[word].astype(np.float64)
    
    def get_negative_word_weight(self, word):
        return self.negative_weights[self.word2index[word]]


class score_simple(generic_document_score):
    method = "simple"

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w, local_counts[w]) for w in tokens])

    def _compute_embedding_vector(self, tokens, **da):
        return np.array([self.get_word_vector(w) for w in tokens])

    def _compute_doc_vector(self, weights, DV, tokens, **da):
        # Build the weight matrix
        W = np.array([weights[w] for w in tokens], dtype=np.float64)
        W = W.reshape(-1, 1)

        # Empty document vector with no tokens, return zero
        if not W.shape[0]:
            dim = self.M.wv.syn0.shape[1]
            return np.zeros((dim,), dtype=np.float64)
        
        # Apply the negative weights
        NV = np.array([self.get_negative_word_weight(w) for w in tokens])
        W *= NV.reshape(-1,1)

        doc_vec = (W * DV).sum(axis=0)
        return L2_norm(doc_vec)


class score_unique(score_simple):
    method = "unique"

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict.fromkeys(tokens, 1.0)

#


class score_simple_TF(score_simple):
    method = "simple_TF"

    def __init__(self, *args, **kwargs):
        super(score_simple, self).__init__(*args, **kwargs)

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

    def get_IDF(self, word):
        if word in self.IDF:
            return self.IDF[word]
        else:
            return 0.0

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w, local_counts[w] * self.get_IDF(w)) for w in tokens])

#


class score_unique_TF(score_simple_TF):
    method = "unique_TF"

    def _compute_item_weights(self, tokens, **da):
        return dict([(w, self.get_IDF(w) * 1.0) for w in tokens])

#


class score_locality_hash(score_unique):
    method = "locality_hash"

    def __init__(self, *args, **kwargs):
        super(score_locality_hash, self).__init__(*args, **kwargs)

        self.f_params = os.path.join(
            kwargs["output_data_directory"],
            "locality_hash_params.pkl")

        params = self.load_params(**kwargs)

        # Build the hash function lookup
        dim = self.M.wv.syn0.shape[1]
        n_bits = int(kwargs['locality_n_bits'])
        alpha = float(kwargs['locality_alpha'])

        R = RBP_hasher(dim, n_bits, alpha)

        # We assume that all locality hashes will be the same, save these
        # params to disk

        for key in ['dim', 'projection_count']:
            if key not in params:
                continue
            print(
                "Checking if locality_hash({}) {}=={}".format(key,
                                                              R.params[key],
                                                              params[key]))
            if R.params[key] != params[key]:
                msg = '''\nLocality-hash config value of {} does
                not match from {} to {}.\nDelete {} to continue.'''
                raise ValueError(msg.format(key, R.params[key],
                                            params[key], self.f_params))

        if 'normals' in params:
            print("Loading locality hash from {}".format(self.f_params))
            R.load(params)
        else:
            joblib.dump(R.params, self.f_params)

        self.RBP_hash = R
        self.WORD_HASH = {}
        for w, v in zip(self.M.wv.index2word, self.M.wv.syn0):
            self.WORD_HASH[w] = self.RBP_hash(v)

    def load_params(self, **kwargs):
        if os.path.exists(self.f_params):
            return joblib.load(self.f_params)
        else:
            return {}

    def _compute_embedding_vector(self, tokens, **da):
        sample_space = self.RBP_hash.sample_space
        DV = np.zeros(shape=(len(tokens), sample_space))
        for i, w in enumerate(tokens):
            for key, val in self.WORD_HASH[w].items():
                DV[i][key] += val
        return DV

    def _compute_doc_vector(self, DV, weights, tokens, **da):

        W = np.array([weights[w] for w in tokens]).reshape(-1, 1)
        doc_vec = (W * DV).sum(axis=0)

        # Locality hash is a probability distribution, so take L1 norm
        doc_vec /= doc_vec.sum()

        # Quick hack
        doc_vec[np.isnan(doc_vec)] = 0

        return doc_vec


#
