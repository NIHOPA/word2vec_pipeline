from document_scores import score_unique_TF
import simple_config
import h5py
import os


class reduced_representation(score_unique_TF):

    method = 'reduced_representation'

    def __init__(self, *args, **kwargs):
        '''
        The reduced representation takes an incremental PCA decomposition
        and adds new negative weights based off the previous components
        of PCA.
        '''

        # Remove the bais to negative_weights
        kwargs["negative_weights"] = {}

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
            ex_var = h5[col]['VX_explained_variance_ratio_'][:]

        bais = config['reduced_representation']['bais_strength']

        self.word_vecs = {}
        for w in self.M.index2word:
            weight = c.dot(self.M[w])
            weight *= bais
            weight *= ex_var
            adjust_v = (weight.reshape(-1, 1) * c).sum(axis=0)
            self.word_vecs[w] = self.M[w] - adjust_v

    def get_word_vector(self, word):
        return self.word_vecs[word]
