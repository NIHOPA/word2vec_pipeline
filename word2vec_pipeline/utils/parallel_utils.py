import joblib


def grouper(iterable, n):
    '''
    Reads ahead n items on an iterator.
    On last pass returns a smaller list with the remaining items.
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

    n_jobs = -1 if FLAG_PARALLEL else 1
    dfunc = joblib.delayed(func)

    with joblib.Parallel(n_jobs=n_jobs) as MP:

        # Yield the whole thing if there isn't a batch_size
        if batch_size is None:
            for z in MP(dfunc(x, *args, **kwargs) for x in INPUT_ITR):
                yield z
            raise StopIteration

        ITR = iter(INPUT_ITR)
        for k, block in enumerate(grouper(ITR, batch_size)):
            for z in MP(dfunc(x, *args, **kwargs) for x in block):
                yield z
