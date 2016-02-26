import h5py, os
import numpy as np
import sklearn.decomposition
from sklearn.decomposition import DictionaryLearning

h5 = h5py.File("collated/document_scores.h5",'r+')
method = "unique"
X = np.vstack(h5[method][key] for key in h5[method])

#from gensim.models.word2vec import Word2Vec
#f_features = "collated/w2v.h5"
#clf = Word2Vec.load(f_features)
#X1 = clf.syn0


DLP = DictionaryLearning(verbose=1,
                         alpha=1.0,
                         split_sign=True,
                         n_components=1000, n_jobs=30)
print "Starting transformation"
DLP.fit(X)

output_method = method + "_DLP"

g = h5.require_group(output_method)

for key in h5[method]:
    X = h5[method][key][:]
    if key in g: del g[key]
    Y = DLP.transform(X)
    g[key] = Y
    print key, Y.shape

    





