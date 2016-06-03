import numpy as np
import h5py
import os, itertools, collections
from tqdm import tqdm

import joblib
import simple_config
from sklearn.cluster import SpectralClustering

from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import cosine_similarity

def subset_iterator(X, m, repeats = 1):
    '''
    Iterates over array X in chunks of m, repeat number of times.
    Each time the order of the repeat is randomly generated.
    '''
    
    N,dim = X.shape
    progress = tqdm(total=repeats*int(N/m))

    for i in range(repeats):
        
        indices = np.random.permutation(N)
        sub_array_n = N//m
        
        for idx in np.array_split(indices,N//m):
            yield X[idx][:]
            progress.update()

    progress.close()


def cosine_affinity(X):
    epsilon = 1e-8
    S = cosine_similarity(X)
    S[S>1] = 1.0 # Rounding problems
    S += 1 + epsilon

    # Sanity checks
    assert(not (S<0).any())
    assert(not np.isnan(S).any())
    assert(not np.isinf(S).any())
    
    return S

def check_h5_item(h5, name, **check_args):
    # Returns True if we need to compute h5[name] and h5[name].attr[key] != val
    
    if name not in h5:
        return True
    
    attrs = h5[name].attrs

    for key, val in check_args.items():
        if (key not in attrs) or (attrs.get(key) != val):
            del h5[name]
            return True

    return False       
        
class cluster_object(object):
    '''
    Helper class to represent all the constitute parts of a clustering
    '''
    def __init__(self):

        config = simple_config.load("metacluster")

        self.subcluster_m = int(config["subcluster_m"])
        self.subcluster_pcut = float(config["subcluster_pcut"])
        self.subcluster_repeats = int(config["subcluster_repeats"])
        self.subcluster_kn = int(config["subcluster_kn"])

        config_score = simple_config.load("score")

        self.f_h5_docvecs = os.path.join(
            config_score["output_data_directory"],
            config_score['document_scores']["f_db"],
        )

        self.f_h5_centroids = os.path.join(
            config["output_data_directory"],
            config["f_centroids"],
        )
        
        score_method = config['score_method']
        text_column  = config['score_column']

        self._load_data(self.f_h5_docvecs, score_method, text_column)

    def _load_data(self, f_h5, method, text_column):

        print "Loading document data from", f_h5

        h5 = h5py.File(f_h5,'r')
        g = h5[method][text_column]
        corpus_keys = g.keys()

        # Load the _refs
        self._refs = np.hstack([g[key]["_ref"][:] for key in corpus_keys])

        # Require the _refs to be in order as a sanity check
        if not (np.sort(self._refs) == self._refs).all():
            msg = "WARNING, data out of sort order from _refs"
            raise ValueError(msg)
        
        self.docv = np.vstack([g[k]["V"][:] for k in corpus_keys])
        self.N,self.dim = self.docv.shape


        # Document key lookup
        print "Building doc lookup key"
        self.doc_lookup = {}
        counter = 0
        for key in corpus_keys:
            for offset in range(g[key]["V"].shape[0]):
                self.doc_lookup[counter] = key
                counter += 1
                
        h5.close()

    def _load_embedding(self):
        f_model = "data_embeddings/w2v.gensim"

        print "Loading embedding", f_model
        import gensim.models.word2vec as W2V
        self.W = W2V.Word2Vec.load(f_model)
        
        return self.W


    def compute_centroid_set(self, **kwargs):

        INPUT_ITR = subset_iterator(self.docv,
                                    self.subcluster_m)

        kn = self.subcluster_kn
        clf = SpectralClustering(
            n_clusters=kn,
            affinity="precomputed",
        )

        C = []

        for X in INPUT_ITR:
            # Remove any rows that have zero vectors
            bad_row_idx = ((X**2).sum(axis=1)==0)
            X = X[~bad_row_idx]
            A = cosine_affinity(X)

            labels = clf.fit_predict(A)
    
            # Compute the centroids
            (N,dim) = X.shape
            centroids = np.zeros((kn,dim))

            for i in range(kn):
                idx = labels==i
                mu = X[idx].mean(axis=0)
                mu /= np.linalg.norm(mu)
                centroids[i] = mu
                
            C.append(centroids)

        return np.vstack(C)
    
    def load_centroid_dataset(self, name):
        h5 = h5py.File(self.f_h5_centroids,'r')
        data = h5[name][:]
        h5.close()
        return data       

    def compute_meta_centroid_set(self, **kwargs):

        C = self.load_centroid_dataset("subcluster_centroids")
        print "Intermediate clusters", C.shape

        # By eye, it looks like the top 60%-80% of the
        # remaining clusters are stable...

        nc = int(self.subcluster_pcut*self.subcluster_kn)
        clf = SpectralClustering(n_clusters=nc,affinity="precomputed")
    
        S = cosine_affinity(C)
        labels = clf.fit_predict(S)

        meta_clusters = []
        meta_cluster_size = []
        for i in range(labels.max()+1):
            idx = labels==i
            mu = C[idx].mean(axis=0)
            mu /= np.linalg.norm(mu)
            meta_clusters.append(mu)
            meta_cluster_size.append(idx.sum())

        return meta_clusters

    def compute_meta_labels(self, **kwargs):

        meta_clusters = self.load_centroid_dataset("meta_centroids")
        n_clusters = meta_clusters.shape[0]

        msg = "Assigning {} labels over {} documents."
        print msg.format(n_clusters, self.N)

        dist = cdist(self.docv, meta_clusters, metric='cosine')
        labels = np.argmin(dist,axis=1)
        
        print "Label distribution: ", collections.Counter(labels)
        return labels

    def docv_centroid_spread(self, **kwargs):
        meta_clusters = self.load_centroid_dataset("meta_centroids")
        meta_labels   = self.load_centroid_dataset("meta_labels")
        n_clusters = meta_clusters.shape[0]

        mu, std, min = [], [], []
        for i in range(n_clusters):
            idx  = meta_labels==i
            X    = self.docv[idx]
            dot_prod = X.dot(meta_clusters[i])
            mu.append ( dot_prod.mean() )
            std.append( dot_prod.std() )
            min.append( dot_prod.min() )

        stats = np.array([mu,std,min])
        return stats

    def describe_clusters(self, **kwargs):

        W = self._load_embedding()

        meta_clusters = self.load_centroid_dataset("meta_centroids")
        n_clusters = meta_clusters.shape[0]
        
        # Find the closest items to each centroid
        all_words = []
        
        for i in range(n_clusters):
            v = meta_clusters[i]

            dist = W.syn0.dot(v)
            idx = np.argsort(dist)[::-1][:10]

            words = [W.index2word[i].replace('PHRASE_','') for i in idx]

            all_words.append(u' '.join(words))

        return np.array(all_words)

if __name__ == "__main__":

    config = simple_config.load("metacluster")

    CO = cluster_object()
    f_h5 = CO.f_h5_centroids

    if not os.path.exists(f_h5):
        h5 = h5py.File(f_h5,'w')   
        h5.close()
        
    h5 = h5py.File(f_h5,'r+')

    keys = ["subcluster_kn", "subcluster_pcut", "subcluster_m", "subcluster_repeats"]
    args = dict([(k,config[k]) for k in keys])

    def compute_func(name, func, dtype=None, **kwargs):

        if check_h5_item(h5, name, **args):
            print "Computing", name
            result = func(**kwargs)

            if dtype in [str,unicode]:
                dt = h5py.special_dtype(vlen=unicode)
                h5.require_dataset(name,shape=result.shape,dtype=dt)
                for i,x in enumerate(result):
                    h5[name][i] = x
            else:
                h5[name] = result    
            
            for k in args:
                h5[name].attrs.create(k,args[k])

    compute_func("subcluster_centroids", CO.compute_centroid_set)
    compute_func("meta_centroids", CO.compute_meta_centroid_set)
    compute_func("meta_labels",    CO.compute_meta_labels)
    compute_func("docv_centroid_spread", CO.docv_centroid_spread)
    compute_func("describe_clusters", CO.describe_clusters,dtype=str)

    print h5['describe_clusters'][:]

    

exit()

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################

class cluster_object(object):
    '''
    Helper class to represent all the constitute parts of a clustering
    '''
    def __init__(self,h5,
                 document_score_method,
                 cluster_method,
                 target_column):

        self.name_doc = document_score_method
        self.name_cluster = cluster_method
        
        g = h5[document_score_method]

        self.labels = g["clustering"][cluster_method][:]
        self.words = g["nearby_words"][cluster_method][:]
        self.T = g["tSNE"][:]
        self.S = g["similarity"][:]
        self.X = load_document_vectors(target_column)
        h5.close()

        assert(self.X.shape[0] == self.T.shape[0] == self.S.shape[0])

        self._label_iter = np.sort(np.unique(self.labels))
        self.cluster_n = self._label_iter.size


    def reorder(self,idx):
        self.X = self.X[idx]
        self.S = self.S[idx][:,idx]
        self.labels = self.labels[idx]
        self.T = self.T[idx]
        
    def __len__(self):
        return self.X.shape[0]

    def sort_labels(self):
        # Reorder the data so the data is the assigned cluster
        idx = np.argsort(self.labels)
        self.reorder(idx)

    def sort_intra(self):
        master_idx = np.arange(len(self))
        
        for i in self._label_iter:
            cidx = self.labels==i
            Z    = self.X[cidx]
            zmu  = Z.sum(axis=0)
            zmu /= np.linalg.norm(zmu)
            
            dist = Z.dot(zmu)
            dist_idx = np.argsort(dist)
            master_idx[cidx] = master_idx[cidx][dist_idx]

        self.reorder(master_idx)


def close_words(W,X,labels,top_n=6):
    '''
    Find words that are close to each label.
    W is a gensim.word2vec
    X is the document vectors.
    labels are predetermined cluster labels.
    '''

    L = []
    for label in np.unique(labels):
        label_idx = labels==label
        mu = X[label_idx].mean(axis=0)
        
        dist = W.syn0.dot(mu)
        idx = np.argsort(dist)[::-1][:top_n]
        words = [W.index2word[i] for i in idx]
        L.append(' '.join(words))

    # Map unicode to simple ASCII
    L = map(unidecode,L)

    # Remove _PHRASE
    L = map(lambda x:x.replace('PHRASE_',''),L)
   
    return L

'''

    if "tSNE" not in group:
        # Compute the tSNE
        print "Computing tSNE for {}".format(method)
        
        if S is None: S = group["similarity"][:]
        local_embed = TSNE(n_components=2,
                           verbose=1,
                           method='exact',
                           metric='precomputed')

        # tSNE expect distances not similarities
        group["tSNE"] = local_embed.fit_transform(1-S)
    
'''
if __name__ == "__main__":
    
    group.require_group("clustering")
    group.require_group("nearby_words")

    for name in config["clustering_commands"]:

        if name not in group["clustering"]:
            # Only load the similarity matrix if needed
            if S is None : S = group["similarity"][:]
            if W is None : W = CSIM.load_embeddings()

            print "Clustering {} {}".format(method, name)
            
            func = getattr(CSIM,name)
            labels = func(S,X,config[name])

            assert(labels.size == X.shape[0])
            
            if name in group["clustering"]: del group["clustering"][name]
            group["clustering"][name] = labels
            
            L = close_words(W,X,labels)
            if name in group["nearby_words"]: del group["nearby_words"][name]
            group["nearby_words"].create_dataset(name, data=L, dtype='S200')


    # Load the cluster object
    document_score_method = method
    cluster_method = "spectral_clustering"
    
    #cluster_method = "hdbscan_clustering"
    
    C = cluster_object(h5_sim,
                       document_score_method,
                       cluster_method,
                       target_column)

    # Sort by labels first
    C.sort_labels()

    # Reorder the intra-clusters by closest to centroid
    C.sort_intra()
    
    # Plot the heatmap
    import pandas as pd
    import seaborn as sns
    plt = sns.plt

    
    fig = plt.figure(figsize=(9,9))
    print "Plotting tSNE"
    colors = sns.color_palette("hls", C.cluster_n)
                               
    for i in C._label_iter:
        x,y = zip(*C.T[C.labels == i])
        label = 'cluster {}, {}'.format(i,C.words[i])

        plt.scatter(x,y,color=colors[i],label=label)
    plt.title("tSNE doc:{} plot".format(C.name_doc),fontsize=16)
    plt.legend(loc=1,fontsize=12)
    plt.tight_layout()
    
    
    fig = plt.figure(figsize=(12,12))
    print "Plotting heatmap"
    df = pd.DataFrame(C.S, index=C.labels,columns=C.labels)
    labeltick = int(len(df)/50)

    sns.heatmap(df,cbar=False,xticklabels=labeltick,yticklabels=labeltick)
    plt.title("heatmap doc:{} clustering:{}".format(C.name_doc,C.name_cluster),fontsize=16)
    plt.tight_layout()
    #plt.savefig("clustering_{}.png".format(n_clusters))
    plt.show()



    '''
    # Load document score data
    X0 = np.vstack([h5[method][name][:]
                    for name in input_names])
    n_docs, n_dims = X0.shape

    #RUHS = random_unit_hypersphere(n_dims)
    RSS = random_spectral_sampling(X0)

    Y0,Y1 = [],[]
    xplot = []
    for k in range(2,10,1):

        # Test this with some clusters?
        
        clf = skc.KMeans(n_clusters=k, n_jobs=-1)

        clusters0 = clf.fit_predict(X0)
        M = compute_cluster_measure(X0,clusters0)
        _,s,_ = np.linalg.svd(M)

        
        #s /= s.max()
        
        #s = s**2
        #f = s/s.sum()
        #inter_entropy = -(f*np.log(f)).sum()
        #print s.min()
        inter_entropy = s.min()

        s = np.diag(M)
        #s = (1-s)**2
        #f = s/s.sum()
        #intra_entropy = -(f*np.log(f)).sum()
        
        #intra_entropy = s.min()#-(f*np.log(f)).sum()
        intra_entropy = 0
        
        xplot.append(k)
        #Y0.append(intra_entropy-inter_entropy)
        Y0.append(inter_entropy)
        Y1.append(intra_entropy)
        print k,inter_entropy, intra_entropy
        continue
    
    import seaborn as sns
    sns.plt.plot(xplot,Y0,label="inter")
    sns.plt.plot(xplot,Y1,label="intra")
    
    sns.plt.legend()
    sns.plt.show()
    '''
    
    
    '''
        X1 = RSS(n_docs)
        clusters1 = clf.fit_predict(X1)
        C1 = compute_cluster_compactness(X1,clusters1)
        
        #for _ in range(10):
        #    clusters1 = clf.fit_predict(X1)
        #    C1 = compute_cluster_compactness(X1,clusters1)
        print k, C1.sum(), C0.sum()
        #gap = np.log(C1.sum()) - np.log(C0.sum())
        #print gap
        #exit()
    '''
    '''
        exit()

        # Load document score data
        X = np.vstack([h5[method][name][:]
                       for name in saved_input_names])

        # Load the categorical columns
        Y = []
        for name in saved_input_names:
            f_sql = os.path.join(pred_dir,name) + '.sqlite'
            engine = create_engine('sqlite:///'+f_sql)
            df = pd.read_sql_table("original",engine,
                                   columns=[column,])
            y = df[column].values
            Y.append(y)

        Y = np.hstack(Y)

        # Determine the baseline prediction
        y_counts = collections.Counter(Y).values()
        baseline_score = max(y_counts) / float(sum(y_counts))

        # Predict
        scores,errors,pred = categorical_predict(X,Y,method,config)

        text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
        print text.format(method, column,
                          scores.mean(), baseline_score)

        PREDICTIONS[method] = pred
        ERROR_MATRIX[method] = errors

    # Build meta predictor
    META_X = np.hstack([PREDICTIONS[method] for method
                        in config["meta_methods"]])
    
    method = "meta"
    scores,errors,pred = categorical_predict(META_X,Y,method,config)

    text = "Predicting [{}] [{}] {:0.4f} ({:0.4f})"
    print text.format(method, column,
                      scores.mean(), baseline_score)

    PREDICTIONS[method] = pred
    ERROR_MATRIX[method] = errors
    '''

