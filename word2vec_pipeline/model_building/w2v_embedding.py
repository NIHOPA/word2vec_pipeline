import os
from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from tqdm import tqdm

import psutil
CPU_CORES = max(4, psutil.cpu_count())

class iterator_factory(object):

    def __init__(self, func, total=0, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.counter = tqdm(total=total)

    def __iter__(self):
        # print "Starting iteration {}".format(self.count)
        ITR = self.func(*self.args, **self.kwargs)
        for x in ITR:
            yield x
        self.counter.update()


class w2v_embedding(corpus_iterator):

    def __init__(
            self,
            epoch_n,
            skip_gram,
            hierarchical_softmax,
            negative,
            window,
            sample,
            size,
            min_count,            
            *args,
            **kwargs
    ):

        # LOOK INTO THIS OBJECT INHERETANCE?
        super(w2v_embedding, self).__init__(*args, **kwargs)
        
        self.epoch_n = int(epoch_n)

        # sg == skip_gram vs cbow
        sg = int(skip_gram)
        hs = int(hierarchical_softmax)
        negative = int(negative)

        # Input bounds checks
        assert(sg in [0, 1])
        assert(hs in [0, 1])

        if hs and negative:
            msg = "If hierarchical_softmax is used negative must be zero"
            raise ValueError(msg)

        self.clf = Word2Vec(
            workers=CPU_CORES,
            sg=sg,
            hs=hs,
            window=int(window),
            negative=negative,
            sample=float(sample),
            size=int(size),
            min_count=int(min_count),
            iter=int(epoch_n),
        )

    def compute(self, target_column='text'):
        print("Learning the vocabulary")

        ITR = iterator_factory(self.sentence_iterator,
                               total=self.epoch_n + 1,
                               target_column=target_column)

        self.clf.build_vocab(ITR)
        
        print("{} words in vocabulary".format(len(self.clf.wv.index2word)))
        print("Training the features")
        
        self.clf.train(
            ITR,
            total_examples=self.clf.corpus_count,
            epochs=self.clf.iter,
        )

        print("Reducing the features")
        self.clf.init_sims(replace=True)

    def save(self, f_db):
        print("Saving the features")
        self.clf.save(f_db)
