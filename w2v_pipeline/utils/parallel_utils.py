
import joblib

def jobmap(func, INPUT_ITR, FLAG_PARALLEL=False):
    
    n_jobs = -1 if FLAG_PARALLEL else 1
    MP = joblib.Parallel(n_jobs=n_jobs)
    dfunc = joblib.delayed(func)

    ITR = (dfunc(x) for x in INPUT_ITR)

    for x in MP(ITR):
        yield x
