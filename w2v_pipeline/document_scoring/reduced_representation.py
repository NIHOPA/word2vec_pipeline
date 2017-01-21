import numpy as np
from sklearn.decomposition import IncrementalPCA

from utils.mapreduce import corpus_iterator
import simple_config
import h5py
import os


class reduced_representation(corpus_iterator):

    method = 'reduced_representation'

    def __init__(self, *args, **kwargs):
        '''
        The reduced representation takes an incremental PCA decomposition
        for all specified sets of scores.
        '''

        super(reduced_representation, self).__init__(*args, **kwargs)

        config = simple_config.load()['score']
        f_db = os.path.join(
            config["output_data_directory"],
            config["document_scores"]["f_db"]
        )

        self.nc = int(config['reduced_representation']['n_components'])
        self.names = config['reduced_representation']['scores_to_reduce']
        self.h5 = h5py.File(f_db, 'r+')

    def compute(self):

        for name in self.names:

            # Check that the name has been pre-scored before reducing
            if name not in self.h5:
                msg = "Must compute {} before running the reduced representation"
                raise ValueError(msg.format(name))

            print("Reducing {} to dimension {}".format(name, self.nc))

            clf = IncrementalPCA(n_components=self.nc)
            V = self.h5[name]['V'][:]
            VX = clf.fit_transform(V)
            _ref = self.h5[name]['_ref'][:]

            save_name = "{}_reduced".format(name)
            if save_name in self.h5:
                del self.h5[save_name]

            g = self.h5.require_group(save_name)
            g.create_dataset("V", data=VX, compression='gzip')
            g.create_dataset("_ref", data=_ref)

    def save(self):
        self.h5.close()
