import numpy as np

from log_probablity import document_log_probability
from document_scores import score_unique_IDF


class score_Z_weighted(score_unique_IDF, document_log_probability):

    method = "Z_weighted"

    def __init__(self, *args, **kwargs):
        """
        The Z weighted method takes the Z scores from the log_probability
        and uses them as a proxy for an IDF weight. Before use, all Z scores
        are mean-centered and scaled to have unit variance. Values above the
        threshold are set to the threshold value. Typically, the threshold is
        set to zero, making all the highly specific words (Z>0) have the same
        weight. The final weighting is the exponential exp(z/kT) where z is the
        thresholded word-weight.
        """

        score_unique_IDF.__init__(self, *args, **kwargs)
        document_log_probability.__init__(self, *args, **kwargs)

        self.kT = float(kwargs["kT"])
        self.threshold = float(kwargs["threshold"])

        self.weights = {}

        for key, val in self.Z.items():
            z = np.exp(min(self.threshold, val) / self.kT)
            self.weights[key] = z

        # Remove all negative items, weights come from Z
        self.neg_W = {}

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w, self.weights[w]) for w in tokens])
