from gensim.models.word2vec import Word2Vec
from mapreduce import simple_mapreduce

class w2v_embedding(simple_mapreduce):

    def set_iterator_function(self, iter_func):
        self.iter_func = iter_func

    def sentence_iterator(self):
        for item in self.iter_func():
            text,idx,f_sql = item
            yield text.split()

    def ITR(self):
        return self.itr_func()

    def __init__(self,*args,**kwargs):
        super(w2v_embedding, self).__init__(*args,**kwargs)

        self.epoch_n = 1

        self.clf = Word2Vec(workers=8,
                            window=5,
                            negative=5,
                            sample=1e-5,
                            size=200,
                            min_count=30)
        

    def compute(self, config):
        print "Learning the vocabulary"
        ITR = self.sentence_iterator()
        self.clf.build_vocab(ITR)

        print "Training the features"
        for n in range(self.epoch_n):
            print " - Epoch {}".format(n)
            ITR = self.sentence_iterator()
            self.clf.train(ITR)

        print "Reducing the features"
        self.clf.init_sims(replace=True)

        print "Saving the features"
        f_features = config["w2v_embedding"]["f_db"]
        self.clf.save(f_features)



