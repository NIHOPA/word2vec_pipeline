import os
from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from tqdm import tqdm

import psutil
CPU_CORES = psutil.cpu_count()
CPU_CORES = 1

"""
File that performs the word2vec training used to create document embeddings. This uses the gensim python
library for natural language processing.
"""

class iterator_factory(object):
    def __init__(self, func, total=0, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.counter = tqdm(total=total)
            
    def __iter__(self):
        #print "Starting iteration {}".format(self.count)
        ITR = self.func(*self.args,**self.kwargs)
        for x in ITR:
            yield x
        self.counter.update()

class w2v_embedding(corpus_iterator):
    """
    Class to perform the word2vec training on documents imported to the pipeline.
    """

    def __init__(self, *args, **kwargs):
        '''
        Initialize the class, and the word2vec model for training
            args: DOCUMENTATION_UNKNOWN
            kwargs: DOCUMENTATION_UNKNOWN
        '''
        super(w2v_embedding, self).__init__(*args, **kwargs)
        self.epoch_n = int(kwargs["epoch_n"])

        # sg == skip_gram vs cbow
        sg = int(kwargs["skip_gram"])
        hs = int(kwargs["hierarchical_softmax"])
        negative = int(kwargs["negative"])

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
            window=int(kwargs["window"]),
            negative=negative,
            sample=float(kwargs["sample"]),
            size=int(kwargs["size"]),
            min_count=int(kwargs["min_count"]),
            iter=int(kwargs["epoch_n"]),
        )

    def compute(self, **config):
        '''
        Build the vocab for the word2vec model, and run the actual training

        Args:
            config: config file
        '''
        print("Learning the vocabulary")

        ITR = iterator_factory(self.sentence_iterator,
                               total=self.epoch_n+1,
                               target_column=config["target_column"])
        
        self.clf.build_vocab(ITR)
        print("{} words in vocabulary".format(len(self.clf.wv.index2word)))

        print("Training the features")
        #for n in tqdm(range(self.epoch_n)):
        #    # print " - Epoch {}".format(n)
        self.clf.train(
            ITR,
            total_examples=self.clf.corpus_count,
            epochs=self.clf.iter,
        )

        print("Reducing the features")
        self.clf.init_sims(replace=True)

        print("Saving the features")
        out_dir = config["output_data_directory"]
        f_features = os.path.join(out_dir, config["w2v_embedding"]["f_db"])
        self.clf.save(f_features)
