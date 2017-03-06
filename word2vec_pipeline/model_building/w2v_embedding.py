import os
from gensim.models.word2vec import Word2Vec
from utils.mapreduce import corpus_iterator
from tqdm import tqdm

import psutil
CPU_CORES = psutil.cpu_count()


class w2v_embedding(corpus_iterator):

    def __init__(self, *args, **kwargs):
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
        )

    def compute(self, **config):
        print("Learning the vocabulary")

        ITR = self.sentence_iterator(config["target_column"])
        self.clf.build_vocab(ITR)

        print("{} words in vocabulary".format(len(self.clf.index2word)))

        print("Training the features")
        for n in tqdm(range(self.epoch_n)):
            # print " - Epoch {}".format(n)
            ITR = self.sentence_iterator(config["target_column"])
            self.clf.train(ITR)

        print("Reducing the features")
        self.clf.init_sims(replace=True)

        print("Saving the features")
        out_dir = config["output_data_directory"]
        f_features = os.path.join(out_dir, config["w2v_embedding"]["f_db"])
        self.clf.save(f_features)
