from scipy.spatial.distance import cdist
import numpy as np
import simple_config
import os


def spectral_clustering(S, X, config):
    '''
    Computes spectral clustering from an input similarity matrix.
    Returns the labels associated with the clustering.
    '''
    from sklearn.cluster import SpectralClustering

    nk = int(config["n_clusters"])
    clf = SpectralClustering(affinity='cosine', n_clusters=nk)
    return clf.fit_predict(X)


def hdbscan_clustering(S, X, config):
    '''
    Computes H-DBSCAN clustering from an input similarity matrix.
    Returns the labels associated with the clustering.
    '''
    from hdbscan import HDBSCAN

    min_size = config.as_int("min_cluster_size")
    clf = HDBSCAN(min_cluster_size=min_size)
    return clf.fit_predict(X)


def load_embeddings():
    '''
    Loads the gensim word embedding model.
    '''
    config = simple_config.load("embedding")

    from gensim.models.word2vec import Word2Vec

    f_w2v = os.path.join(
        config["output_data_directory"],
        config["w2v_embedding"]["f_db"],
    )

    return Word2Vec.load(f_w2v)


def compute_document_similarity(X):
    '''
    From a matrix of unit distances, computes the cosine similarity
    then changes to the angular distance (for a proper metric).
    '''

    S = cdist(X, X, metric='cosine')
    S -= 1
    S *= -1
    S[S > 1] = 1.0
    S[S < 0] = 0.0

    # Set nan values to zero
    S[np.isnan(S)] = 0

    # Convert to angular distance (a proper metric)
    S = 1 - (np.arccos(S) / np.pi)
    assert(not np.isnan(S).any())
    assert(not np.isinf(S).any())

    return S


'''
class random_unit_hypersphere(object):
    def __init__(self,dim=3):
        self.dim = dim

    def generate_random_unit_hypersphere_point(self,*args):
        return np.random.normal(size=self.dim)

    def __call__(self, n=5):
        func = self.generate_random_unit_hypersphere_point
        INPUT_ITR = itertools.repeat(self.dim,n)
        ITR = itertools.imap(func,INPUT_ITR)
        result = np.array(list(ITR))
        return result

class random_spectral_sampling(object):
    def __init__(self,X):
        dim = X.shape[1]
        U,s,V = np.linalg.svd(X,full_matrices=False)
        s = np.diag(s)

        self.dim = dim//5
        self.U = U[:,:self.dim]
        self.s = s[:self.dim,:self.dim]
        self.V = V[:self.dim,:]

    def __call__(self, n=5):
        print "HI!"
        UX = np.random.uniform(-1,1,size=(n,self.dim))
        UX /= np.linalg.norm(UX,axis=0)
        Z = UX.dot(self.s.dot(self.V))
        Z = (Z.T/np.linalg.norm(Z,axis=1)).T
        return Z


def compute_cluster_means(X,clusters):
    MU = []
    for i in np.arange(clusters.max()+1):
        idx = clusters==i
        mu = X[idx].sum(axis=0)
        mu /= np.linalg.norm(mu)
        MU.append(mu)
    return np.array(MU)

def compute_cluster_measure(X,clusters):
    MU = compute_cluster_means(X,clusters)
    cx = np.arange(clusters.max()+1)
    M = np.zeros((len(cx),len(cx)))
    for i in cx:
        for j in cx:
            idx = clusters==i
            Z = X[idx].dot(MU[j])
            M[i,j] = Z.mean()

    return M

def compute_cluster_compactness(X,clusters):
    # W_k
    compactness = []
    for i in np.arange(clusters.max()+1):
        idx = clusters==i
        mu = X[idx].sum(axis=0)
        mu /= np.linalg.norm(mu)
        delta = mu-X[idx]
        z = np.linalg.norm(delta,axis=1)
        compactness.append(z.mean())
    return np.array(compactness)
'''
