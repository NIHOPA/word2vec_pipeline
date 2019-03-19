"""
Utility functions to assist in parallelizing the operation of the pipeline.
"""

import joblib
from tqdm import tqdm


def grouper(itr, n):
    """
    Reads ahead n items on an iterator. On last pass returns a smaller
    list with the remaining items. Useful for batch processing in parallel.

    Args:
        itr (iterable): a object to iterate over
        n (int): Number of items to read ahead

    Yields:
        list: A list of at least n items from the iterable
    """

    block = []
    while True:
        try:
            block.append(itr.next())
        except StopIteration:
            break
        if len(block) == n:
            yield block
            block = []

    if block:
        yield block


def jobmap(
    func, INPUT_ITR, FLAG_PARALLEL=False, batch_size=None, *args, **kwargs
):
    """
    Function to parallalize the operation of another function
    passed to it.

    Args:
        func (function): Run in parallel on the input
        INPUT_ITR (iterable): Input to be operated on
        FLAG_PARALLEL (bool): Flag to run the functions in parallel
        batch_size (int):
        args: additional arguments passed to the function
        kwargs: additional keyword arguments passed to the function
    """

    n_jobs = -1 if FLAG_PARALLEL else 1
    dfunc = joblib.delayed(func)

    with joblib.Parallel(n_jobs=n_jobs) as MP:

        # Yield the whole thing if there isn't a batch_size
        if batch_size is None:
            for z in MP(dfunc(x, *args, **kwargs) for x in INPUT_ITR):
                yield z
            return

        ITR = iter(INPUT_ITR)
        progress_bar = tqdm()
        for block in grouper(ITR, batch_size):
            MPITR = MP(dfunc(x, *args, **kwargs) for x in block)
            for k, z in enumerate(MPITR):
                yield z
            progress_bar.update(k + 1)
