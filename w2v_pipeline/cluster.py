import numpy as np
import h5py
import os, itertools, collections

import clustering.similarity as CSIM
from utils.os_utils import mkdir

from sklearn.manifold import TSNE

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

    X = np.vstack(h5_score[method][key] for key in keys)
    h5_score.close()
    return X


def reorder_data(idx, X, S, labels):
    return X[idx], S[idx][:,idx], labels[idx]


if __name__ == "__main__":

    import simple_config   

    config = simple_config.load("cluster")
    output_dir = config["output_data_directory"]
    mkdir(output_dir)

    method = 'unique'

    f_sim = os.path.join(output_dir, config["f_cluster"])
    if not os.path.exists(f_sim):
        h5_sim = h5py.File(f_sim,'w')
        h5_sim.close()

    h5_sim = h5py.File(f_sim,'r+')
    group = h5_sim.require_group(method)
    S = None

    if "similarity" not in group:

        # Load the document scores
        X = load_document_vectors()

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

    for name in config["clustering_commands"]:

        if name not in group["clustering"] or config.as_bool("_FORCE"):
            # Only load the similarity matrix if needed
            if S is None: S = group["similarity"][:]

            print "Clustering {} {}".format(method, name)
            
            func = getattr(CSIM,name)
            del group["clustering"][name]
            group["clustering"][name] = func(S,config[name])


    labels = group["clustering"]["spectral_clustering"][:]
    S = group["similarity"][:]
    X = load_document_vectors()
    T = group["tSNE"][:]


    # Reorder the data so the data is the assigned cluster
    idx = np.argsort(labels)
    X, S, labels = reorder_data(idx, X, S, labels)
    T = T[idx,:]  

    # Reorder the intra-clusters by closest to centroid
    n_items = S.shape[0]
    master_idx = np.arange(n_items)
    cluster_n = max(labels)+1

    for cluster_i in range(cluster_n):
        cluster_idx = labels==cluster_i
        Z = X[cluster_idx]
        zmu = Z.sum(axis=0)
        zmu /= np.linalg.norm(zmu)
        
        dist = Z.dot(zmu)
        dist_idx = np.argsort(dist)
        master_idx[cluster_idx] = master_idx[cluster_idx][dist_idx]

    X, S, labels = reorder_data(master_idx, X, S, labels)
    T = T[master_idx,:]
    
    # Plot the heatmap
    import pandas as pd
    import seaborn as sns
    plt = sns.plt

    fig = plt.figure(figsize=(9,9))
    print "Plotting tSNE"
    colors = sns.color_palette("hls", cluster_n)
                               
    for i in range(cluster_n):
        x,y = zip(*T[labels == i])
        #label = 'cluster {}, {}'.format(i,WORDS_NEARBY[i])
        label = 'cluster {}'.format(i)
        plt.scatter(x,y,color=colors[i],label=label)
    plt.title("tSNE plot",fontsize=16)
    plt.legend(loc='best',fontsize=16)
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure(figsize=(12,12))
    print "Plotting heatmap"
    df = pd.DataFrame(S, index=labels,columns=labels)
    labeltick = int(len(df)/50)

    sns.heatmap(df,cbar=False,xticklabels=labeltick,yticklabels=labeltick)
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

