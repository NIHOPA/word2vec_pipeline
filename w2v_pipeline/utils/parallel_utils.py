import itertools
import joblib

def grouper(iterable, n, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip(*args)

def jobmap(func, INPUT_ITR, FLAG_PARALLEL=False, batch_size=None,
           *args, **kwargs):
    
    n_jobs = -1 if FLAG_PARALLEL else 1
    dfunc = joblib.delayed(func)

    with joblib.Parallel(n_jobs=n_jobs) as MP:

        # Yield the whole thing if there isn't a batch_size
        if batch_size is None:
            for z in MP(dfunc(x,*args,**kwargs) for x in INPUT_ITR):
                yield z
            raise StopIteration

        for block in grouper(INPUT_ITR,batch_size):
            raise NotImplementedError("batch_size has bug, do not use")
            for z in MP(dfunc(x,*args,**kwargs) for x in block):
                yield z
