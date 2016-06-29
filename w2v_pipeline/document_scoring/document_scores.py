import collections, itertools, os
import numpy as np
import pandas as pd
import h5py

from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from locality_hashing import RBP_hasher


class document_scores(corpus_iterator):

    def __init__(self,*args,**kwargs):
        super(document_scores, self).__init__(*args,**kwargs)

        # Check if we need to load term_frequency data
        methods = kwargs['methods']

        f_w2v = os.path.join(
            kwargs["embedding"]["output_data_directory"],
            kwargs["embedding"]["w2v_embedding"]["f_db"],
        )

        # Load the model from disk
        self.M = Word2Vec.load(f_w2v)
        self.shape = self.M.syn0.shape
        
        # Build the dictionary
        self.methods = kwargs["methods"]
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.index2word,range(vocab_n)))

        if "locality_hash" in methods:
            pass
        
        # Set parallel option
        self._PARALLEL = kwargs["_PARALLEL"]

    def _compute_item_weights(self, **da):
        # Method has not been defined, raise error here
        msg = "UNKNOWN w2v weights {}".format(self.method)
        raise KeyError(msg)

    def _compute_embedding_vector(self, **da):
        # Method has not been defined, raise error here
        msg = "UNKNOWN w2v embedding {}".format(self.method)
        raise KeyError(msg)

    def L1_norm(self, doc_vec):
        # Renormalize onto the hypersphere
        doc_vec /= np.linalg.norm(doc_vec)
        
        # Sanity check, L1 norm if tokens exist
        if doc_vec.any():
            assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
        else:
            dim = self.M.syn0.shape[1]
            doc_vec = np.zeros(dim,dtype=float)
            
        return doc_vec

    def _compute_doc_vector(self, **da):
        method = self.current_method
        
        if method in ["locality_hash"]:
            doc_vec = np.array(da['DV']).sum(axis=0)

            # Only keep track if the hypercube corner is occupied
            # doc_vec[doc_vec>0] = 1

            # Renormalize onto the hypersphere
            doc_vec /= np.linalg.norm(doc_vec)

            # Quick hack
            doc_vec[ np.isnan(doc_vec) ] = 0

        else:
            msg = "UNKNOWN w2v method '{}'".format(method)
            raise KeyError(msg)

        return doc_vec

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

        for self.current_method in self.methods:
            print "Scoring {}".format(self.current_method)
            ITR = itertools.imap(self.score_document, self)
            
            data = []
            for result in ITR:
                data.append(result)

            data = np.array(data)
            
            df = pd.DataFrame(data=data,
                              columns=["V","_ref","table_name","f_sql"])
            df.set_index("_ref",inplace=True)

            self.save(config, df)

    def save(self, config, df):

        method = self.current_method

        print "Saving the scored documents"
        out_dir = config["output_data_directory"]
        f_db = os.path.join(out_dir, config["document_scores"]["f_db"])

        # Create the h5 file if it doesn't exist
        if not os.path.exists(f_db):
            h5 = h5py.File(f_db,'w')
        else:
            h5 = h5py.File(f_db,'r+')

        g1  = h5.require_group(method)
        
        for key_table,df2 in df.groupby("table_name"):
            g2 = g1.require_group(key_table)
            
            for key_sql,df3 in df2.groupby("f_sql"):

                # Save into the group of the base file name
                name = '.'.join(os.path.basename(key_sql).split('.')[:-1])

                # Save the data array
                print "Saving", method, key_table, name, df3["V"].shape
                V = np.array(df3["V"].tolist())

                # Save the _ref numbers
                _ref = np.array(df3.index.tolist())

                # Sanity check on sizes
                all_sizes = set([x.shape for x in V])
                if len(all_sizes) != 1:
                    msg = "Method {} failed, sizes differ {}"
                    raise ValueError(msg.format(name, all_sizes))

                if name in g2: del g2[name]

                g3 = g2.require_group(name)
                g3.create_dataset("V",data=V,compression='gzip')
                g3.create_dataset("_ref",data=_ref)


        h5.close()


class generic_document_score(document_scores):

    def __init__(self,*args,**kwargs):

        # This line will be phased out eventually
        kwargs['methods'] = []
        
        super(generic_document_score, self).__init__(*args,**kwargs)

    def compute(self, config):

        assert(self.method is not None)
        
        print "Scoring {}".format(self.method)
        ITR = itertools.imap(self.score_document, self)
            
        data = []
        for result in ITR:
            data.append(result)

        data = np.array(data)
            
        df = pd.DataFrame(data=data,
                          columns=["V","_ref","table_name","f_sql"])
        df.set_index("_ref",inplace=True)
        self.save(config, df)


    def save(self, config, df):

        print "Saving the scored documents"
        out_dir = config["output_data_directory"]
        f_db = os.path.join(out_dir, config["document_scores"]["f_db"])

        # Create the h5 file if it doesn't exist
        if not os.path.exists(f_db):
            h5 = h5py.File(f_db,'w')
        else:
            h5 = h5py.File(f_db,'r+')

        g1  = h5.require_group(self.method)
        
        for key_table,df2 in df.groupby("table_name"):
            g2 = g1.require_group(key_table)
            
            for key_sql,df3 in df2.groupby("f_sql"):

                # Save into the group of the base file name
                name = '.'.join(os.path.basename(key_sql).split('.')[:-1])

                # Save the data array
                print "Saving", self.method, key_table, name, df3["V"].shape
                V = np.array(df3["V"].tolist())

                # Save the _ref numbers
                _ref = np.array(df3.index.tolist())

                # Sanity check on sizes
                all_sizes = set([x.shape for x in V])
                if len(all_sizes) != 1:
                    msg = "Method {} failed, sizes differ {}"
                    raise ValueError(msg.format(name, all_sizes))

                if name in g2: del g2[name]

                g3 = g2.require_group(name)
                g3.create_dataset("V",data=V,compression='gzip')
                g3.create_dataset("_ref",data=_ref)


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
        return self.L1_norm(doc_vec)

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
        
        # Build the hash function lookup
        dim = self.M.syn0.shape[1]
        n_bits = int(kwargs['locality_n_bits'])
        alpha = float(kwargs['locality_alpha'])
            
        self.RBP_hash = RBP_hasher(dim,n_bits,alpha)
        self.WORD_HASH = {}
        for w,v in zip(self.M.index2word, self.M.syn0):
            self.WORD_HASH[w] = self.RBP_hash(v)

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

        doc_vec = self.L1_norm(doc_vec)
        
        # Quick hack
        doc_vec[ np.isnan(doc_vec) ] = 0

        return doc_vec


##################################################################################
