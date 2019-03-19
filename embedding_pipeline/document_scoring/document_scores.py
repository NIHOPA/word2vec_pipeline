"""
Score each document imported to the pipeline using a gensim word2vec model.
The generic_document_score class is inherited by each other scoring class.
Word scores can be weighted in the config file, which affects clustering
and classification.
"""

import collections
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .utils.data_utils import load_w2vec
from .utils.os_utils import save_h5, get_h5save_object

import logging

logger = logging.getLogger(__name__)


def L2_norm(doc_vec):
    """
    Renormalize document vector onto the hypersphere.

    Args:
        doc_vec (numpy array): Document vector

    Returns:
        doc_vec: Normalized document vector
    """

    # Renormalize onto the hypersphere
    doc_vec /= np.linalg.norm(doc_vec)

    # Sanity check, L2 norm and set to zeros if not
    if doc_vec.any():
        assert np.isclose(1.0, np.linalg.norm(doc_vec))
    else:
        logger.warning("Warning L2 norm not satisifed (0-vector returned)")
        doc_vec = np.zeros(doc_vec.shape)

    return doc_vec


def token_counts(tokens, size_mb=1):
    """
    Returns a count for the number of times a token appears in a list.
    bounter is slower here since we aren't counting a large enough corpus.
    """
    return collections.Counter(tokens)


# ----------------------------------------------------------------------------


class generic_document_score(object):

    """
    Class to score documents with word2vec model, using scoring method
    specified. Each scoring method has its own class associated with it
    that inherits generic_document_score()
    """

    def __init__(self, downsample_weights=None, *args, **kwargs):
        """
        Initialize the class, loading the word2vec model. If any words are
        given a downsample weight then they are applied here.

        Args:
            *args: DOCUMENTATION_UNKNOWN
            **kwargs: DOCUMENTATION_UNKNOWN
        """

        # Load the model from disk
        self.M = load_w2vec()

        # Build the dictionary, and a mapping from word2index
        self.shape = self.M.wv.syn0.shape
        self.vocab = dict(zip(self.M.wv.index2word, range(self.shape[0])))

        self.DSW = np.ones(shape=len(self.vocab), dtype=float)

        for word, weight in downsample_weights.items():

            if not self.check_word_vector(word):
                msg = "Downsample word '{}' not found in dictionary"
                logger.warning(msg.format(word))
                continue

            vec = self.get_word_vector(word)
            scale = np.exp(-float(weight) / 1.0 * self.M.wv.syn0.dot(vec))
            scale = np.clip(scale, 0, 1)

            self.DSW *= scale

        # Make sure nothing has been set yet
        self.V = self._ref = None

    def _empty_vector(self):
        return np.zeros((self.shape[1],), dtype=float)

    def check_word_vector(self, word):
        # Reuturns True/False if the word vector is in the vocab
        return word in self.vocab

    def get_word_vector(self, word):
        return self.M[word].astype(np.float64)

    def get_word_vectors(self, ws):
        return np.array([self.get_word_vector(w) for w in ws])

    def get_downsample_word_weights(self, ws):
        return np.array([self.DSW[self.vocab[w]] for w in ws])

    def get_counts(self, counter_object, keys):
        return np.array([counter_object[x] for x in keys])

    def get_tokens_from_text(self, text):
        tokens = text.split()

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.vocab]

        if not valid_tokens:
            return []
            # raise Warning("No valid tokens in document!")

        return valid_tokens

    def __call__(self, text):
        raise NotImplementedError

    def save(self, data, f_csv, f_db):
        """
        Takes in a (data) dictionary of _ref:doc_vec and saves it to an
        h5 file.  Save only to the basename of the file (f_csv), saves to
        (f_db).
        """

        _refs = sorted(data.keys())
        V = np.array([data[r] for r in _refs])

        # Sanity check, should not have any NaNs
        assert not np.isnan(V).any()

        # Set the size explictly as a sanity check
        size_n, dim_V = V.shape

        g = get_h5save_object(f_db, self.method)
        gx = g.require_group(os.path.basename(f_csv))

        save_h5(gx, "V", V)
        save_h5(gx, "_ref", _refs)

    def compute_vectors(self, tokens, need_counts=False, need_IDF=False):

        token_counter = token_counts(tokens)
        words = list(token_counter)
        item = {"words": words}

        item["n"] = self.get_downsample_word_weights(words)
        item["W"] = self.get_word_vectors(words)

        if need_counts:
            item["C"] = self.get_counts(token_counter, words)

        if need_IDF:
            item["idf"] = self.get_IDF_weights(words)

        return item


# ----------------------------------------------------------------------------


class IDF_document_score(generic_document_score):
    def __init__(
        self,
        output_data_directory=None,
        term_document_frequency=None,
        *args,
        **kwargs
    ):

        super(IDF_document_score, self).__init__(
            output_data_directory=output_data_directory, *args, **kwargs
        )

        assert term_document_frequency is not None

        f_csv = os.path.join(
            output_data_directory, term_document_frequency["f_db"]
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


# ----------------------------------------------------------------------------


class score_simple(generic_document_score):
    method = "simple"

    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        if not tokens:
            return self._empty_vector()

        v = self.compute_vectors(tokens, need_counts=True)
        C, n, W = v["C"], v["n"], v["W"]

        return L2_norm(((C * n) * W.T).sum(axis=1))


class score_unique(generic_document_score):
    method = "unique"

    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        if not tokens:
            return self._empty_vector()

        v = self.compute_vectors(tokens)
        n, W = v["n"], v["W"]

        return L2_norm((n * W.T).sum(axis=1))


class score_simple_IDF(IDF_document_score):
    method = "simple_IDF"

    def __call__(self, text):
        tokens = self.get_tokens_from_text(text)
        if not tokens:
            return self._empty_vector()

        v = self.compute_vectors(tokens, need_counts=True, need_IDF=True)
        C, n, W, idf = v["C"], v["n"], v["W"], v["idf"]

        return L2_norm(((idf * C * n) * W.T).sum(axis=1))


class score_unique_IDF(IDF_document_score):
    method = "unique_IDF"

    def __call__(self, text):
        tokens = set(self.get_tokens_from_text(text))
        if not tokens:
            return self._empty_vector()

        v = self.compute_vectors(tokens, need_IDF=True)
        n, W, idf = v["n"], v["W"], v["idf"]

        return L2_norm(((idf * n) * W.T).sum(axis=1))


class score_IDF_common_component_removal(score_unique_IDF):
    method = "IDF_common_component_removal"

    def __call__(self, text):
        """
        Adapts one of the ideas from the paper "A SIMPLE BUT TOUGH-TO-BEAT 
        BASELINE FOR SENTENCE EMBEDDINGS", 

        https://openreview.net/forum?id=SyK00v5xx

        by subtracting off the main principal component from the data.
        
        https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        """

        tokens = set(self.get_tokens_from_text(text))
        if not tokens:
            return self._empty_vector()

        v = self.compute_vectors(tokens, need_counts=True, need_IDF=True)
        C, n, W, idf = v["C"], v["n"], v["W"], v["idf"]

        WX = (n * idf) * W.T / C

        # If there is only one vector, no need to compute PCA
        if len(tokens) > 1:

            # Do not center the data, subtract out the common discouse vector
            clf = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            u = clf.fit(WX).components_
            WX -= WX.dot(u.T) * u

        return L2_norm(WX.sum(axis=1))
