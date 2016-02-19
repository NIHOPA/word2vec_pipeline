
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
