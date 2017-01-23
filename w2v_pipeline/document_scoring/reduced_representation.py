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
        for k, (weight, vec) in enumerate(zip(ex_var, c)):
            name = "__negative_weight_{}".format(k)
            self.neg_W[name] = weight * bais
            self.neg_vec[name] = vec
