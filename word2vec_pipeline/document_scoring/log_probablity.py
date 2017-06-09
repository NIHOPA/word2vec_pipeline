from utils.mapreduce import corpus_iterator
import numpy as np
import os

import pandas as pd
import tqdm
import h5py
import scipy.stats

import simple_config
from utils.data_utils import load_w2vec


def compute_partition_stats(UE):

    # Remove the largest element (the energy of self interaction)
    UE = np.sort(UE)[:-1]
    n_words = len(UE)

    return UE.sum() / n_words


def compute_stats(X, data, prefix):
    # Compute various measures of central tendency

    name = "{}_".format(prefix)

    data[name + "mu"] = np.mean(X)
    data[name + "std"] = np.std(X)
    data[name + "skew"] = scipy.stats.skew(X)
    data[name + "kurtosis"] = scipy.stats.kurtosis(X)


class document_log_probability(corpus_iterator):

    table_name = 'log_prob'
    method = 'document_log_probability'

    def __init__(self, *args, **kwargs):
        '''
        Computes various measures of central tendency of a document.
        For Z_X scores, the raw word tokens are summed over the partition
        function. For I_X scores, the same statistics are computed over
        the similarity of all word pairs for words with top 10% Z values.
        This will precompute the partition function if it doesn't exist.
        '''
        cfg_embed = simple_config.load()["embedding"]
        cfg_score = simple_config.load()["score"]

        f_w2v = os.path.join(
            cfg_embed["output_data_directory"],
            cfg_embed["w2v_embedding"]["f_db"],
        )

        f_partition_function = os.path.join(
            cfg_embed["output_data_directory"],
            cfg_score["document_log_probability"]["f_partition_function"],
        )

        if not os.path.exists(f_partition_function):
            self.create_partition_function(f_w2v, f_partition_function)

        self.Z = self.load_partition_function(f_partition_function)
        self.scores = []

        val = cfg_score["document_log_probability"]["intra_document_cutoff"]
        self.intra_document_cutoff = float(val)

        self.model = load_w2vec()

    def energy(self, a, b):
        return a.dot(b)

    def create_partition_function(self, f_w2v, f_h5):
        print("Building the partition function")

        # Load the model from disk
        M = load_w2vec()

        words = M.wv.index2word
        ZT = []
        INPUT_ITR = tqdm.tqdm(words)

        # Compute the partition function for each word
        for w in INPUT_ITR:
            UE = self.energy(M.wv.syn0, M[w])
            z = compute_partition_stats(UE)
            ZT.append(z)

        # Save the partition function to disk
        # (special care needed for h5py unicode strings)
        dt = h5py.special_dtype(vlen=unicode)

        with h5py.File(f_h5, 'w') as h5:

            h5.create_dataset("words", (len(words),),
                              dtype=dt,
                              data=[w.encode('utf8') for w in words])

            h5.attrs['vocab_N'] = len(words)
            h5['Z'] = ZT

    def load_partition_function(self, f_h5):
        '''
        The partition function is a dictionary of the
        Standardized (zero-mean, unit-variance) Z scores are returned
        that were precomputed over the corpus embedding.

        '''

        with h5py.File(f_h5, 'r') as h5:
            words = h5["words"][:]
            Z = h5['Z'][:]

        # Standardize Z scores
        Z = (Z - Z.mean()) / Z.std()

        # Sanity check that the number of words matches what was saved
        assert(len(words) == len(Z))

        return dict(zip(words, Z))

    def __call__(self, row):
        '''
        Compute partition function stats over each document.
        '''
        text = row['text']

        stat_names = [
            'Z_mu', 'Z_std', 'Z_skew', 'Z_kurtosis',
            'I_mu', 'I_std', 'I_skew', 'I_kurtosis',
        ]
        stats = {}
        for key in stat_names:
            stats[key] = 0.0

        # Only keep words that are defined in the embedding
        valid_tokens = [w for w in text.split() if w in self.Z]

        # Take only the unique words in the document
        all_tokens = np.array(list(set(valid_tokens)))

        if len(all_tokens) > 3:

            # Possibly clip the values here as very large Z don't contribute
            doc_z = np.array([self.Z[w] for w in all_tokens])
            compute_stats(doc_z, stats, "Z")

            # Take top x% most descriptive words
            z_sort_idx = np.argsort(doc_z)[::-1]
            z_cut = max(int(self.intra_document_cutoff * len(doc_z)), 3)

            important_index = z_sort_idx[:z_cut]
            sub_tokens = all_tokens[important_index]
            doc_v = np.array([self.model[w] for w in sub_tokens])
            upper_idx = np.triu_indices(doc_v.shape[0], k=1)
            dist = np.dot(doc_v, doc_v.T)[upper_idx]

            compute_stats(dist, stats, "I")

        stats['_ref'] = row['_ref']
        return stats

    def reduce(self, stats):
        self.scores.append(stats)

    def save(self, config):

        out_dir = config["output_data_directory"]
        f_h5 = os.path.join(out_dir,
                            config["document_log_probability"]["f_db"])

        df = pd.DataFrame(self.scores)

        with h5py.File(f_h5, 'w') as h5:
            h5['_ref'] = df['_ref'].astype(int)
            del df['_ref']

            for key in df.columns:
                h5[key] = df[key].astype(float)
