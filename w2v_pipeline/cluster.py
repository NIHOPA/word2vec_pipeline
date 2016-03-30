import clustering
import numpy as np
import pandas as pd
import h5py
import os, glob, itertools, collections
import sklearn.cluster as skc

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
    '''
    W_k
    '''
    compactness = []
    for i in np.arange(clusters.max()+1):
        idx = clusters==i
        mu = X[idx].sum(axis=0)
        mu /= np.linalg.norm(mu)
        delta = mu-X[idx]
        z = np.linalg.norm(delta,axis=1)
        compactness.append(z.mean())
    return np.array(compactness)


if __name__ == "__main__":

    import simple_config
    config = simple_config.load("clustering")

    f_h5 = config["f_db_scores"]
    h5 = h5py.File(f_h5,'r')

    methods  = config["methods"]
    pred_dir = config["predict_target_directory"]

    input_glob  = os.path.join(pred_dir,'*')
    input_files = glob.glob(input_glob)
    input_names = ['.'.join(os.path.basename(x).split('.')[:-1])
                   for x in input_files]

    ITR = iter(methods)

    #for method in ITR:
    method = ITR.next()
    print method
    
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

