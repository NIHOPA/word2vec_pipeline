from __future__ import division

import h5py, os
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.misc import factorial
import seaborn as sns

def random_hypersphere_point(dim):
    pts = np.random.normal(size=dim)
    return pts / np.linalg.norm(pts)

from gensim.models.doc2vec import Doc2Vec
f_features = "collated/d2v.h5"
clf = Doc2Vec.load(f_features)

low_cut = 1000
high_cut = 2000

target = "_bio.sqlite"
names = sorted([w for w in clf.docvecs.doctags if target in w])
X_doc2vec = np.array([clf.docvecs[w] for w in names])
X_doc2vec = X_doc2vec[low_cut:high_cut]
dist_doc2vec = pdist(X_doc2vec,metric='cosine')

target = "_compbio.sqlite"
names2 = sorted([w for w in clf.docvecs.doctags if target in w])
X_doc2vec2 = np.array([clf.docvecs[w] for w in names2])
X_doc2vec2 = X_doc2vec2[low_cut:high_cut, :]

dist_doc2vec2 = pdist(X_doc2vec2,metric='cosine')



from gensim.models.word2vec import Word2Vec
f_features = "collated/w2v.h5"
clf = Word2Vec.load(f_features)
X_word = clf.syn0[low_cut:high_cut]
dist_word = pdist(X_word,metric='cosine')
n,dim = X_word.shape
print X_word.shape

rand_pts = [random_hypersphere_point(dim) for _ in xrange(n)]
dist_rand = pdist(rand_pts,metric='cosine')

h5 = h5py.File("collated/document_scores.h5",'r')
X_doc = h5["unique"]["PLoS_bio"]
dist_doc = pdist(X_doc[low_cut:high_cut],metric='cosine')

sns.distplot(dist_word, label="word tokens")
sns.distplot(dist_doc, label="w2vec document sums")
sns.distplot(dist_doc2vec, label="bio doc2vec tokens")
sns.distplot(dist_doc2vec2, label="compbio doc2vec tokens")
sns.distplot(dist_rand, label="random points")
sns.plt.xlim(0,1.5)
sns.plt.tight_layout()
sns.plt.legend()

idx = np.triu_indices(X_doc2vec.shape[0],k=1)
inter_doc  = cdist(X_doc2vec, X_doc2vec2,metric='cosine')[idx]
intra_doc1 = cdist(X_doc2vec, X_doc2vec,metric='cosine')[idx]
intra_doc2 = cdist(X_doc2vec2, X_doc2vec2,metric='cosine')[idx]

sns.plt.figure()
sns.distplot(inter_doc, label="bio vs compbio")
sns.distplot(intra_doc1, label="bio vs bio")
sns.distplot(intra_doc2, label="compbio vs compbio")
sns.plt.tight_layout()
sns.plt.legend()


sns.plt.show()

