from gensim.models.word2vec import Word2Vec
from utils.mapreduce import simple_mapreduce
import numpy as np
import os

import pandas as pd
import sqlalchemy


class document_log_probability(simple_mapreduce):

    table_name = 'log_prob'

    def __init__(self, f_db, minimum_sentence_length, *args, **kwargs):
        '''
        Computes the average log probability of a document.
        # (maybe not...) requires that the embedding model has hs=1, negative=0.
        '''

        self.window = int(kwargs['embedding']['w2v_embedding']['window'])

        f_w2v = os.path.join(
            kwargs["embedding"]["output_data_directory"],
            kwargs["embedding"]["w2v_embedding"]["f_db"],
        )
        
        # Load the model from disk
        self.M = M = Word2Vec.load(f_w2v)
        self.shape = self.M.syn0.shape

        N = self.shape[0]
        self.Z = {}
        #self.word2index = dict(zip(*[range(N), M.index2word]))
        for w in self.M.index2word:
            v = M[w]
            UE = np.dot(self.M.syn0, v)
            self.Z[w] = np.exp(UE-1.0).sum()

        '''
        if M.hs != 1:
            msg = "hierarchical_softmax=1 is required for log_probablity"
            raise ValueError(msg)

        if M.negative:
            msg = "negative=0 is required for log_probablity"
            raise ValueError(msg)
        '''

        self.min_sent = int(minimum_sentence_length)
        self.scores = {}        

    def __call__(self,item):
        '''
        Compute the local partition function for each word.
        '''
        _ref = item[1]

        # Debug line
        #if _ref>20:return [(_ref,-400)] + item[1:]

        sents = item[0].split('\n')        

        # Only keep sentences that are this long
        #sents = [s for s in sents if len(s.split()) >= self.min_sent]
        sents_n = len(sents)

        doc_p = []

        for sent in sents:
            tokens = [w for w in sent.split() if w in self.M]

            if len(tokens) < self.min_sent:
                continue
            
            vecs = np.array([self.M[w] for w in tokens])
            n = vecs.shape[0]
            N = self.shape[0]

            sent_p = []

            for i in range(n):
                left_idx  = max(0,i-self.window+1)
                right_idx = min(n,i+self.window+1)

                inner_vecs = np.vstack([vecs[left_idx:i], vecs[i+1:right_idx]])

                if not inner_vecs.size:
                    continue

                inner_vecs = inner_vecs.sum(axis=0)

                # These should already be unit normalized
                #inner_vecs /= np.linalg.norm(inner_vecs)

                uv = np.dot(inner_vecs, vecs[i])
                prob = np.exp(uv-1.0)

                prob /= self.Z[tokens[i]]

                # Multiple the probability by the uniform distribution
                prob *= N
                
                sent_p.append(prob)

            # Take the average of the sentence (if any tokens)
            avg_sent_p = np.array(sent_p).mean()
            doc_p.append( avg_sent_p )

            # Debug print statement
            #if avg_sent_p < 1.0:
            #    print sent, avg_sent_p

        if len(doc_p):
            avg_doc_p = np.array(doc_p).mean()
        else:
            avg_doc_p = 0.0

        return [(_ref,avg_doc_p)] + item[1:]

    def reduce(self,(_ref,score)):
        self.scores[_ref] = score

    def save(self, config):

        df = pd.DataFrame(self.scores.items(),
                          columns=["_ref","log_prob"])

        # Scale to the most negative
        #df.log_prob -= df.log_prob.max()

        out_dir = config["output_data_directory"]
        f_sql = os.path.join(out_dir,
                             config["document_log_probability"]["f_db"])
        
        engine = sqlalchemy.create_engine('sqlite:///'+f_sql)

        df.to_sql(self.table_name,
                  engine,
                  index=False,
                  if_exists='replace')
