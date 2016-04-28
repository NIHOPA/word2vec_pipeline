import numpy as np
import h5py
import os, itertools, collections

import clustering.similarity as CSIM
from utils.os_utils import mkdir
from unidecode import unidecode
from sklearn.manifold import TSNE

class cluster_object(object):
    '''
    Helper class to represent all the constitute parts of a clustering
    '''
    def __init__(self,h5,document_score_method, cluster_method):

        self.name_doc = document_score_method
        self.name_cluster = cluster_method
        
        g = h5[document_score_method]

        self.labels = g["clustering"][cluster_method][:]
        self.words = g["nearby_words"][cluster_method][:]
        self.T = g["tSNE"][:]
        self.S = g["similarity"][:]
        self.X = load_document_vectors()
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

def load_document_vectors():

    config_score = simple_config.load("score")

    f_h5 = os.path.join(
        config_score["output_data_directory"],
        config_score["document_scores"]["f_db"],
    )
    h5_score = h5py.File(f_h5,'r') 
      
    print "Loading the document scores", h5_score

    keys = h5_score[method].keys()

    if config["command_whitelist"]:
        keys = [k for k in keys if k in config["command_whitelist"]]
        print "Only computing over", keys

    X = np.vstack(h5_score[method][key] for key in keys)
    h5_score.close()
    return X



if __name__ == "__main__":

    import simple_config   

    config = simple_config.load("cluster")
    output_dir = config["output_data_directory"]
    mkdir(output_dir)

    method = 'unique'
    
    f_sim = os.path.join(output_dir, config["f_cluster"])

    if config.as_bool("_FORCE"):
        os.remove(f_sim)
    
    
    if not os.path.exists(f_sim):
        h5_sim = h5py.File(f_sim,'w')
        h5_sim.close()

    h5_sim = h5py.File(f_sim,'r+')
    group = h5_sim.require_group(method)
    S = None
    W = None

    # Load the document scores
    X = load_document_vectors()

    if "similarity" not in group:

        # Compute and save the similarity matrix
        print "Computing the similarity matrix"
        
        # Save the similarity matrix
        S = CSIM.compute_document_similarity(X)
        group["similarity"] = S

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
    
    C = cluster_object(h5_sim, document_score_method, cluster_method)

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

