
class simple_mapreduce(object):
    
    def __init__(self,*args,**kwargs):
        # Set any function arguments for calling
        self.kwargs = kwargs
            
    def __call__(self, x):
        raise NotImplementedError

    def reduce(self,*x):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

class corpus_iterator(simple_mapreduce):

    def set_iterator_function(self, iter_func, *args):
        self.iter_func = iter_func
        self.iter_args = args

    def __iter__(self):
        for x in self.iter_func(*self.iter_args):
            yield x

    def sentence_iterator(self):
        for item in self:
            text,idx,f_sql = item
            yield text.split()
            



    
    
