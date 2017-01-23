from sklearn.decomposition import IncrementalPCA
import numpy as np

from document_scores import score_unique_TF
import simple_config
import h5py
import os


class reduced_representation(score_unique_TF):

    method = 'reduced_representation'

    def __init__(self, *args, **kwargs):
        '''
        The reduced representation takes an incremental PCA decomposition
        and [WORK IN PROGRESS]
        '''

        super(reduced_representation, self).__init__(*args, **kwargs)

        config = simple_config.load()['score']
        f_db = os.path.join(
            config["output_data_directory"],
            config["document_scores"]["f_db"]
        )


        with h5py.File(f_db, 'r') as h5:

            # Make sure the the column has a value
            col = config['reduced_representation']['rescored_command']
            assert(col in h5)

            # Make sure the VX has been computed
            assert("VX" in h5[col])
            c = h5[col]['VX_components_'][:]

        M  = self.M.syn0
        MX = M.dot(c.T).dot(c)

        self.word_reconstruct = {}
        
        for w,i in self.word2index.items():
            self.word_reconstruct[w] = 1 - M[i].dot(MX[i])
            if self.word_reconstruct[w] < 0.5:
                self.word_reconstruct[w] = 0
                

    def get_RECON(self, word):
        if word in self.word_reconstruct:
            return self.word_reconstruct[word]
        else:
            return 0.0        

    def _compute_item_weights(self, tokens, **da):
        return dict([(w, self.get_IDF(w)*self.get_RECON(w)) for w in tokens])

