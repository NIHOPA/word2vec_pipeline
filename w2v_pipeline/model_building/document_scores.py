import collections, itertools, os, ast
import numpy as np
import pandas as pd
import h5py

from gensim.models.word2vec import Word2Vec
from mapreduce import corpus_iterator

import tqdm

class document_scores(corpus_iterator):

    def __init__(self,*args,**kwargs):
        super(document_scores, self).__init__(*args,**kwargs)

         # Load the model from disk
        self.M = Word2Vec.load(kwargs["f_w2v"])       
        self.shape = self.M.syn0.shape
        
        # Build total counts
        self.counts = {}
        for key,val in self.M.vocab.items():
            self.counts[key] = val.count

        # Build the dictionary
        self.methods = kwargs["methods"]
        vocab_n = self.shape[0]
        self.word2index = dict(zip(self.M.index2word,range(vocab_n)))

        # Set parallel option
        self._PARALLEL = kwargs["_PARALLEL"]

    def score_document(self, item):

        text,meta,idx,f_sql = item
        tokens = text.split()

        # Find out which tokens are defined
        valid_tokens = [w for w in tokens if w in self.M]
        local_counts = collections.Counter(valid_tokens)
        tokens = set(valid_tokens)
        method = self.current_method

        if not tokens:
            msg = "Document has no valid tokens! This is problem."
            raise ValueError(msg)


        # If scoring function requires meta, convert it
        if method in ["pos_split"]:
            meta = ast.literal_eval(meta)

        # Lookup the weights (model dependent)
        if method in ["unique","pos_split"]:
            weights = dict.fromkeys(tokens, 1.0)
        elif method in ["simple","svd_stack"]:
            weights = dict([(w,local_counts[w]) for w in tokens])
        elif method in ["TF_IDF","kSVD"]:
            weights = dict([(w,IDF[w]*c) 
                            for w,c in local_counts.items()])
        else:
            msg = "UNKNOWN w2v method {}".format(method)
            raise KeyError(msg)

        # Lookup the embedding vector
        if method in ["unique","simple","TF_IDF","svd_stack"]:
            DV = np.array([self.M[w] for w in tokens])
        elif method in ["kSVD"]:
            word_idx = [self.word2index[w] for w in tokens]
            DV = [self.kSVD_gamma[n] for n in word_idx]
        elif method in ["pos_split"]:

            known_tags = ["N","ADJ","V"]
            dim = self.M.syn0.shape[1]
            pos_vecs = {}
            pos_totals = {}
            for pos in known_tags:
                pos_vecs[pos] = np.zeros((dim,),dtype=float)
                pos_totals[pos] = 0

            POS = meta["POS"]
            ordered_tokens = [t for t in text.split()]
            for token,pos in zip(text.split(),meta["POS"]):
                if token in valid_tokens and pos in known_tags:

                    # This is the "unique" weights
                    if token in pos_vecs:
                        continue
                    pos_vecs[pos]   += self.M[token]
                    pos_totals[pos] += 1

            # Normalize
            for pos in known_tags:
                pos_vecs[pos] /= pos_totals[pos]
            
        else:
            msg = "UNKNOWN w2v method '{}'".format(method)
            raise KeyError(msg)

        # Sum all the vectors with their weights
        if method in ["simple","unique"]:
            # Build the weight matrix
            W  = np.array([weights[w] for w in tokens]).reshape(-1,1)
            DV = np.array(DV)

            doc_vec = (W*DV).sum(axis=0)

            # Renormalize onto the hypersphere
            doc_vec /= np.linalg.norm(doc_vec)

            # Sanity check, L1 norm
            assert(np.isclose(1.0, np.linalg.norm(doc_vec)))
        elif method in ["pos_split"]:
            
            # Concatenate
            doc_vec = np.hstack([pos_vecs[pos] for pos in known_tags])

        elif method in ["svd_stack"]:
            # Build the weight matrix
            W  = np.array([weights[w] for w in tokens]).reshape(-1,1)
            DV = np.array(DV)

            n = 2
            _U,_s,_V = np.linalg.svd(DV)
            doc_vec = np.hstack([np.hstack(_V[:n]), _s[:n]])
        else:
            msg = "UNKNOWN w2v method '{}'".format(method)
            raise KeyError(msg)
        

        return doc_vec,idx,f_sql

    def compute(self, config):
        '''
        if self._PARALLEL:
            import multiprocessing
            MP = multiprocessing.Pool()
            ITR = MP.imap(self.score_document, self.iter_func())
        else:
            ITR = itertools.imap(self.score_document, self.iter_func())
        '''

        for self.current_method in self.methods:
            print "Scoring {}".format(self.current_method)
                    
            ITR = itertools.imap(self.score_document, self)
            
            data = []
            for result in tqdm.tqdm(ITR):
                data.append(result)

            df = pd.DataFrame(data=data,
                              columns=["V","idx","f_sql"])

            self.save(config, df)

    def save(self, config, df):

        method = self.current_method

        print "Saving the scored documents"
        f_db = config["document_scores"]["f_db"]

        # Create the h5 file if it doesn't exist
        if not os.path.exists(f_db):
            h5 = h5py.File(f_db,'w')
        else:
            h5 = h5py.File(f_db,'r+')

        for key,data_group in df.groupby("f_sql"):

            # Save into the group of the base file name
            name = '.'.join(os.path.basename(key).split('.')[:-1])
            
            g  = h5.require_group(method)

            V = np.array(data_group["V"].tolist())
            print "Saving", name, method, V.shape
            
            if name in g:
                del g[name]

            g.create_dataset(name,
                             data=V,
                             compression='gzip')

        h5.close()
