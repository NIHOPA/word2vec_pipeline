import joblib
from tqdm import tqdm


"""
Utility functions to assist in parallelizing the operation of the pipeline.
"""

#DOCUMENTATION_UNKNOWN
#not sure what this does
def grouper(iterable, n):
    '''
    Reads ahead n items on an iterator. On last pass returns a smaller list with the remaining items.

    Args:
        iterable: an iterable list
        n: an integer, the number of items to read ahead
    '''
    block = []
    while True:
        try:
            block.append(iterable.next())
        except StopIteration:
            break
        if len(block) == n:
            yield block
            block = []

    yield block


def jobmap(func, INPUT_ITR, FLAG_PARALLEL=False, batch_size=None,
           *args, **kwargs):
    '''
    Function to parallalize the operation of another function that is passed to it on the given input
    Args:
        func: Function that is run in parallel on the input
        INPUT_ITR: Iterable list of documents that are operated on in parallel
        FLAG_PARALLEL: Boolean flag to run the functions in parallel
        batch_size: An integer
        args: additional arguments
        kwargs: additional arguments
    '''

    n_jobs = -1 if FLAG_PARALLEL else 1
    dfunc = joblib.delayed(func)

    with joblib.Parallel(n_jobs=n_jobs) as MP:

        # Yield the whole thing if there isn't a batch_size
        if batch_size is None:
            for z in MP(dfunc(x, *args, **kwargs)
                        for x in INPUT_ITR):
                yield z
            raise StopIteration

        ITR = iter(INPUT_ITR)
        progress_bar = tqdm()
        for block in grouper(ITR, batch_size):
            MPITR = MP(dfunc(x, *args, **kwargs) for x in block)
            for k,z in enumerate(MPITR):
                yield z
            progress_bar.update(k+1)
