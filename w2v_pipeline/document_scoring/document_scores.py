import collections, itertools, os, joblib
import numpy as np
import h5py
import tqdm

from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from locality_hashing import RBP_hasher


def L2_norm(doc_vec):
    # Renormalize onto the hypersphere
    doc_vec /= np.linalg.norm(doc_vec)
        
    # Sanity check, L2 norm and set to zeros if not
    if doc_vec.any():
        assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
    else:
        print "Warning L2 norm not satisifed (zero-vector returned)"
        doc_vec = np.zeros(doc_vec.shape)
        
    return doc_vec

def touch_h5(f_db):
    # Create the h5 file if it doesn't exist
    if not os.path.exists(f_db):
        h5 = h5py.File(f_db,'w')
    else:
        h5 = h5py.File(f_db,'r+')
    return h5

##################################################################################    

class generic_document_score(corpus_iterator):

    def __init__(self,*args,**kwargs):

        super(generic_document_score, self).__init__(*args,**kwargs)

        f_w2v = os.path.join(
            kwargs["embedding"]["output_data_directory"],
            kwargs["embedding"]["w2v_embedding"]["f_db"],
        )

        # Load the model from disk
        self.M = Word2Vec.load(f_w2v)
        self.shape = self.M.syn0.shape
        
        # Build the dictionary
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.index2word,range(vocab_n)))
        
        # Set parallel option (currently does nothing)
        self._PARALLEL = kwargs["_PARALLEL"]

    def _compute_item_weights(self, **da):
        msg = "UNKNOWN w2v weights {}".format(self.method)
        raise KeyError(msg)

    def _compute_embedding_vector(self, **da):
        msg = "UNKNOWN w2v embedding {}".format(self.method)
        raise KeyError(msg)

    def _compute_doc_vector(self, **da):
        msg = "UNKNOWN w2v doc vec {}".format(self.method)
        raise KeyError(msg)

    def score_document(self, item):

        text = unicode(item[0])
        idx  = item[1]
        other_args = item[2:]
        
        tokens = text.split()

        # Document args
        da = {}

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.M]
        
        da["local_counts"] = collections.Counter(valid_tokens)
        da["tokens"] = set(valid_tokens)
        
        if not da["tokens"]:
            msg = "Document has no valid tokens! This is probably a problem."
            print msg
            #raise ValueError(msg)

        da["weights"] = self._compute_item_weights(**da)
        da['DV'] = self._compute_embedding_vector(**da)
        da['doc_vec'] = self._compute_doc_vector(**da)
        
        # Sanity check, should not have any NaNs
        assert(not np.isnan(da['doc_vec']).any())

        return [da['doc_vec'],idx,] + other_args
        

    def compute(self, config):
        # Save each block (table_name, f_sql) as its own
        

        assert(self.method is not None)

        print "Scoring {}".format(self.method)
        
        for block in self:
            
            ITR = itertools.imap(self.score_document, tqdm.tqdm(block))
            data = zip(*(list(ITR)))
            f_sql = data[3][0]
            table_name = data[2][0]
            V = np.array(data[0])
            _ref = np.array(data[1])
            
            self.save(config, V, _ref, f_sql, table_name)
            

    
    def save(self, config, V, _ref, f_sql, table_name):
        
        # Set the size explictly as a sanity check
        size_n, dim_V = V.shape
        
        # print "Saving the scored documents"
        out_dir = config["output_data_directory"]
        f_db = os.path.join(out_dir, config["document_scores"]["f_db"])

        h5 = touch_h5(f_db)
        g1 = h5.require_group(self.method)
        g2 = g1.require_group(table_name)

        # Save into the group of the base file name
        name = '.'.join(os.path.basename(f_sql).split('.')[:-1])
        
        # Save the data array
        print "Saving {} {} {} ({})".format(self.method, table_name, name, size_n)
                
        # Clear the dataset if it already exists
        if name in g2: del g2[name]

        g3 = g2.require_group(name)
        g3.create_dataset("V",data=V,compression='gzip',shape=(size_n,dim_V))
        g3.create_dataset("_ref",data=_ref,shape=(size_n,))

        h5.close()


##################################################################################
        
class score_simple(generic_document_score):
    method = "simple"

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w,local_counts[w]) for w in tokens])

    def _compute_embedding_vector(self, tokens, **da):
        return np.array([self.M[w] for w in tokens])    

    def _compute_doc_vector(self, weights, DV, tokens, **da):
        # Build the weight matrix
        W  = np.array([weights[w] for w in tokens]).reshape(-1,1)
        
        doc_vec = (W*DV).sum(axis=0)
        return L2_norm(doc_vec)

class score_unique(score_simple):
    method = "unique"

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict.fromkeys(tokens, 1.0)

##################################################################################    

class score_simple_TF(score_simple):
    method = "simple_TF"

    def __init__(self,*args,**kwargs):
        super(score_simple, self).__init__(*args,**kwargs)
        
        f_db = os.path.join(
            kwargs['output_data_directory'],
            kwargs['term_frequency']['f_db']
        )
        if not os.path.exists(f_db):
            msg = "{} not computed yet, needed for TF methods!"
            raise ValueError(msg.format(f_db))

        import sqlalchemy
        engine = sqlalchemy.create_engine('sqlite:///'+f_db)

        import pandas as pd
        IDF = pd.read_sql_table("term_document_frequency",engine)
        IDF = dict(zip(IDF["word"].values, IDF["count"].values))
            
        self.corpus_N = IDF.pop("")
            
        # Compute the IDF
        for key in IDF:
            IDF[key] = np.log(float(self.corpus_N)/(IDF[key]+1))
        self.IDF = IDF
        

    def _compute_item_weights(self, local_counts, tokens, **da):
        return dict([(w,local_counts[w]*self.IDF[w]) for w in tokens])
    
##################################################################################

class score_unique_TF(score_simple_TF):
    method = "unique_TF"

    def _compute_item_weights(self, tokens, **da):
        return dict([(w,self.IDF[w]*1.0) for w in tokens])

##################################################################################

class score_locality_hash(score_unique):
    method = "locality_hash"

    def __init__(self,*args,**kwargs):
        super(score_locality_hash, self).__init__(*args,**kwargs)

        self.f_params = os.path.join(
            kwargs["output_data_directory"],
            "locality_hash_params.pkl")

        params = self.load_params(**kwargs)

        # Build the hash function lookup
        dim = self.M.syn0.shape[1]
        n_bits = int(kwargs['locality_n_bits'])
        alpha = float(kwargs['locality_alpha'])

        R = RBP_hasher(dim,n_bits,alpha)

        # We assume that all locality hashes will be the same, save these params to disk
        
        for key in ['dim', 'projection_count']:
            if key not in params: continue
            print "Checking if locality_hash({}) {}=={}".format(key, R.params[key], params[key])
            if R.params[key] != params[key]:
                msg = "\nLocality-hash config value of {} does not match from {} to {}.\nDelete {} to continue."
                raise ValueError(msg.format(key, R.params[key], params[key], self.f_params))

        if 'normals' in params:
            print "Loading locality hash from {}".format(self.f_params)
            R.load(params)
        else:
            joblib.dump(R.params, self.f_params)

        self.RBP_hash = R        
        self.WORD_HASH = {}
        for w,v in zip(self.M.index2word, self.M.syn0):
            self.WORD_HASH[w] = self.RBP_hash(v)


    def load_params(self, **kwargs):
        if os.path.exists(self.f_params):
            return joblib.load(self.f_params)
        else:
            return {}

    def _compute_embedding_vector(self, tokens, **da):
        sample_space = self.RBP_hash.sample_space
        DV = np.zeros(shape=(len(tokens), sample_space))
        for i,w in enumerate(tokens):
            for key,val in self.WORD_HASH[w].items():
                DV[i][key] += val
        return DV

    def _compute_doc_vector(self, DV, weights, tokens, **da):

        W  = np.array([weights[w] for w in tokens]).reshape(-1,1)
        doc_vec = (W*DV).sum(axis=0)

        # Locality hash is a probability distribution, so take L1 norm
        doc_vec /= doc_vec.sum()
        
        # Quick hack
        doc_vec[ np.isnan(doc_vec) ] = 0

        return doc_vec


##################################################################################
