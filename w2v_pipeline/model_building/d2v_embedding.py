import os
from gensim.models.doc2vec import Doc2Vec
from utils.mapreduce import corpus_iterator

import psutil
CPU_CORES = psutil.cpu_count()

import gensim.models
assert gensim.models.doc2vec.FAST_VERSION > -1


class d2v_embedding(corpus_iterator):

    def __init__(self, *args, **kwargs):
        super(d2v_embedding, self).__init__(*args, **kwargs)

        self.epoch_n = int(kwargs["epoch_n"])

        self.clf = Doc2Vec(workers=CPU_CORES,
                           window=int(kwargs["window"]),
                           negative=int(kwargs["negative"]),
                           sample=float(kwargs["sample"]),
                           size=int(kwargs["size"]),
                           min_count=int(kwargs["min_count"])
                           )

    def compute(self, config):
        print("Learning the vocabulary")

        ITR = self.labelized_sentence_iterator()
        self.clf.build_vocab(ITR)

        print("Training the features")
        for n in range(self.epoch_n):
            print(" - Epoch {}".format(n))
            ITR = self.labelized_sentence_iterator()
            self.clf.train(ITR)

        print("Reducing the features")
        self.clf.init_sims(replace=True)

        print("Saving the features")
        out_dir = config["output_data_directory"]
        f_features = os.path.join(out_dir, config["d2v_embedding"]["f_db"])
        self.clf.save(f_features)
