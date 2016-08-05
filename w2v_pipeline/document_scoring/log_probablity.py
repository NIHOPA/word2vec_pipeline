from gensim.models.word2vec import Word2Vec
from utils.mapreduce import simple_mapreduce
import numpy as np
import os

import pandas as pd
import sqlalchemy
import tqdm, h5py

class document_log_probability(simple_mapreduce):

    table_name = 'log_prob'

    def __init__(self, f_db, minimum_sentence_length, *args, **kwargs):

        '''
        Computes the average log probability of a document.
        Break document up into sentences, then breaks document into windows.
        Each window computes the probability that a phase is formed, scaled
        by the center word's partition function. Will precompute the
        partition function if it doesn't exist.
        '''

        # Internal temperature (not configurable yet)
        self.kT = 1.0
        
        self.window = int(kwargs['embedding']['w2v_embedding']['window'])

        f_w2v = os.path.join(
            kwargs["embedding"]["output_data_directory"],
            kwargs["embedding"]["w2v_embedding"]["f_db"],
        )
        
        # Load the model from disk
        self.M = M = Word2Vec.load(f_w2v)
        self.shape = self.M.syn0.shape

        f_partition_function = os.path.join(
            kwargs["embedding"]["output_data_directory"],
            kwargs["f_partition_function"],
        )
        if not os.path.exists(f_partition_function):
            self.create_partition_function(f_partition_function)

        self.Z = self.load_partition_function(f_partition_function)

        self.min_sent = int(minimum_sentence_length)
        self.scores = {}

    def energy(self, a, b):
        return a.dot(b)

    def unnorm_prob(self, v):
        return np.exp((v-1.0)/self.kT)

    def probability(self, word, energy):
        return self.unnorm_prob(energy) / self.Z[word]

    def window_probability(self, word, word_vec, context_vec):
        u = self.energy(context_vec, word_vec)
        p = self.probability(word, u)

        # Multiply the probability by the uniform distribution
        # for eaiser human interperation.
        vocab_N = self.shape[0]
        return p*vocab_N

    def create_partition_function(self, f_h5):
        print "Building the partition function"

        words = self.M.index2word
        Z = []
        
        for w in tqdm.tqdm(words):
            v  = self.M[w]
            UE = self.energy(self.M.syn0, v)
            Z.append( self.unnorm_prob(UE).sum() )

        dt = h5py.special_dtype(vlen=unicode)

        with h5py.File(f_h5,'w') as h5:
                       
            h5.create_dataset("words", (len(words),),
                              dtype=dt,
                              data=[w.encode('utf8') for w in words])
            h5['Z'] = Z

        
    def load_partition_function(self, f_h5):
        with h5py.File(f_h5,'r') as h5:
            return dict(zip(h5["words"][:], h5["Z"][:]))

    def sentence_iterator(self, sent):
        tokens = [w for w in sent.split() if w in self.M]
        vecs = np.array([self.M[w] for w in tokens])
        sent_N = vecs.shape[0]

        if len(tokens) < self.min_sent:
            raise StopIteration

        for i in range(sent_N):
            left_idx  = max(0,i-self.window+1)
            right_idx = min(sent_N,i+self.window+1)
            inner_vecs = np.vstack([vecs[left_idx:i], vecs[i+1:right_idx]])

            context_vec  = inner_vecs.sum(axis=0)
            context_vec /= np.linalg.norm(context_vec)

            word = tokens[i]
            word_vec = vecs[i]

            yield word, word_vec, context_vec



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

            sent_p = []
            window_itr = self.sentence_iterator(sent)
            for word, word_vec, context_vec in window_itr:
                
                p = self.window_probability(word, word_vec, context_vec)
                sent_p.append(p)

            if not sent_p:
                continue

            sent_p = np.array(sent_p)

            # Take the average of the sentence (if any tokens)
            doc_p.append( sent_p.mean() )

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
