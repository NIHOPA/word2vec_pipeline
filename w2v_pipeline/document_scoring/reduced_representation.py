import numpy as np

#from log_probablity import document_log_probability
from document_scores import score_unique_TF

class reduced_representation(score_unique_TF):

    method = 'reduced_representation'

    def __init__(self, *args, **kwargs):
        '''
        The reduced representation takes an incremental PCA decomposition
        for all specified sets of scores.
        '''

        score_unique_TF.__init__(self, *args, **kwargs)

        print "HI!"
        exit()

        self.kT = float(kwargs["kT"])
        self.threshold = float(kwargs["threshold"])

        self.weights = {}

        # min_val = np.array(self.Z.values()).min()
        # print min_val

        for key, val in self.Z.items():
            z = np.exp(min(self.threshold, val) / self.kT)
            self.weights[key] = z

        # Remove all negative items, weights come from Z
        self.neg_W = {}

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w, self.weights[w]) for w in tokens])
