import h5py, os
import numpy as np
import sklearn.decomposition

h5 = h5py.File("collated/document_scores.h5",'r+')

method = "unique"
X = np.vstack(h5[method][key] for key in h5[method])


PCA = sklearn.decomposition.PCA()
print "Starting transformation"
PCA.fit(X)

output_method = method + "_PCA_rotation"

g = h5.create_group(output_method)

for key in h5[method]:
    X = h5[method][key][:]
    g[key] = PCA.transform(X)
    print key
    





