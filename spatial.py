from __future__ import division

import h5py, os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.misc import factorial

def random_hypersphere_point(dim):
    pts = np.random.normal(size=dim)
    return pts / np.linalg.norm(pts)

h5 = h5py.File("collated/document_scores.h5",'r')
X2 = h5["unique"]["PLoS_bio"][:]
dist2 = pdist(X2[1000:2000],metric='cosine')

#exit()
from gensim.models.word2vec import Word2Vec
f_features = "collated/w2v.h5"
clf = Word2Vec.load(f_features)
X1 = clf.syn0


X1 = X1[1000:2000]
dist1 = pdist(X1,metric='cosine')
n,dim = X1.shape
print X1.shape

rand_pts = [random_hypersphere_point(dim) for _ in xrange(n)]
dist_rand = pdist(rand_pts,metric='cosine')


import seaborn as sns
sns.distplot(dist1, label="word tokens")
sns.distplot(dist2, label="w2vec document sums")
sns.distplot(dist_rand, label="random points")
sns.plt.legend()
sns.plt.show()

