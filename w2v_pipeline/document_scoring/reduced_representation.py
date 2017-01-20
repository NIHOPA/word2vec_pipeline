import numpy as np
from sklearn.decomposition import IncrementalPCA

#from log_probablity import document_log_probability
from document_scores import score_unique_TF

class reduced_representation(score_unique_TF):

    method = 'reduced_representation'

    def __init__(self, *args, **kwargs):
        '''
        The reduced representation takes an incremental PCA decomposition
        for all specified sets of scores. Right now it is hard-coded to be unique_TF
        '''

        super(reduced_representation,self).__init__(*args, **kwargs)
        
        config = kwargs['reduced_representation']
        self.clf = IncrementalPCA(n_components=int(config['n_components']))
        self.batch = []
        self.batch_n = 2000

    def process_doc_vec(self, v):
        self.batch.append(v)
        if len(self.batch) >= self.batch_n:
            self.clf.partial_fit(self.batch)
            self.batch = []

    def _compute_doc_vector(self, weights, DV, tokens, **da):

        func = super(reduced_representation,self)._compute_doc_vector
        doc_vec = func(weights,DV,tokens,**da)

        self.process_doc_vec(doc_vec)
        return doc_vec
    
    def save(self):

        # Fit the remaining batch
        self.clf.partial_fit(self.batch)

        # Transform the batch
        self.V = self.clf.transform(self.V)

        # Run the standard save
        score_unique_TF.save(self)

