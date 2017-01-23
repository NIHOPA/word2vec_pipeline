import collections
import itertools
import os
import joblib
import simple_config
import numpy as np
import h5py
import pandas as pd

from tqdm import tqdm

from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from locality_hashing import RBP_hasher
from sklearn.decomposition import IncrementalPCA


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


def touch_h5(f_db):
    # Create the h5 file if it doesn't exist
    if not os.path.exists(f_db):
        h5 = h5py.File(f_db, 'w')
    else:
        h5 = h5py.File(f_db, 'r+')
    return h5

#


class generic_document_score(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(generic_document_score, self).__init__(*args, **kwargs)

        config_embed = simple_config.load("embedding")
        f_w2v = os.path.join(
            config_embed["output_data_directory"],
            config_embed["w2v_embedding"]["f_db"],
        )

        # Load the model from disk
        self.M = Word2Vec.load(f_w2v)
        self.shape = self.M.syn0.shape

        # Build the dictionary
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.index2word, range(vocab_n)))

        # Set parallel option (currently does nothing)
        self._PARALLEL = kwargs["_PARALLEL"]

        # Load the negative weights
        if "negative_weights" in kwargs:
            neg_W = kwargs["negative_weights"]
            self.neg_W = dict((k, float(v)) for k, v in neg_W.items())
            self.neg_vec = dict((k, self.get_word_vector(k))
                                for k, v in neg_W.items())
        else:
            self.neg_W = {}
            self.neg_vec = {}

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
            msg = "Document has no valid tokens! This is probably a problem."
            print(msg)
            # raise ValueError(msg)

        da["weights"] = self._compute_item_weights(**da)
        da['DV'] = self._compute_embedding_vector(**da)
        da['doc_vec'] = self._compute_doc_vector(**da)

        # Sanity check, should not have any NaNs
        assert(not np.isnan(da['doc_vec']).any())

        row['doc_vec'] = da['doc_vec']
        return row

    def compute(self):
        # Save each block (table_name, f_sql) as its own

        assert(self.method is not None)
        print("Scoring {}".format(self.method))

        ITR = tqdm(itertools.imap(self.score_document, self))

        data = {}
        for k, row in enumerate(ITR):
            data[int(row['_ref'])] = row['doc_vec']

        self._ref = sorted(data.keys())
        self.V = np.vstack([data[k] for k in self._ref])

    def save(self):

        assert(self.V is not None)
        assert(self._ref is not None)

        # Set the size explictly as a sanity check
        size_n, dim_V = self.V.shape

        # print "Saving the scored documents"
        config_score = simple_config.load("score")
        f_db = os.path.join(
            config_score["output_data_directory"],
            config_score["document_scores"]["f_db"]
        )

        h5 = touch_h5(f_db)

        # Clear the dataset if it already exists
        if self.method in h5:
            del h5[self.method]

        g = h5.require_group(self.method)

        # Save the data array
        print("Saving {} ({})".format(self.method, size_n))

        g.create_dataset("V", data=self.V, compression='gzip')
        g.create_dataset("_ref", data=self._ref)

        # Compute the reduced representation if required
        if self.compute_reduced:
            nc = self.reduced_n_components
            clf = IncrementalPCA(n_components=nc)

            msg = "Performing PCA on {}, ({})->({})"
            print(msg.format(self.method, self.V.shape[1], nc))

            VX = clf.fit_transform(self.V)
            g.create_dataset("VX", data=VX, compression='gzip')
            g.create_dataset("VX_explained_variance_ratio_",
                             data=clf.explained_variance_ratio_)
            g.create_dataset("VX_components_",
                             data=clf.components_)

        h5.close()

    def get_word_vector(self, word):
        return self.M[word]

    def get_negative_word_vector(self, word):
        return self.neg_vec[word]


class score_simple(generic_document_score):
    method = "simple"

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w, local_counts[w]) for w in tokens])

    def _compute_embedding_vector(self, tokens, **da):
        return np.array([self.get_word_vector(w) for w in tokens])

    def _compute_doc_vector(self, weights, DV, tokens, **da):
        # Build the weight matrix
        W = np.array([weights[w] for w in tokens]).reshape(-1, 1)

        # Empty document vector with no tokens, return zero
        if not W.shape[0]:
            dim = self.M.syn0.shape[1]
            return np.zeros((dim,))

        # Apply the negative weights
        # This needs to be multipled across the unit sphere so
        # it "spreads" across and not just applies it to a single word.

        for neg_word, neg_weight in self.neg_W.items():
            neg_vec = self.get_negative_word_vector(neg_word)
            neg_scale = np.exp(-neg_weight * DV.dot(neg_vec))
            # Don't oversample, max out weights to unity
            neg_scale[neg_scale > 1] = 1.0
            W = W * neg_scale.reshape(-1, 1)

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

        score_config = simple_config.load("score")
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
        dim = self.M.syn0.shape[1]
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
        for w, v in zip(self.M.index2word, self.M.syn0):
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
