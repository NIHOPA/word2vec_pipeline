import itertools, h5py, os
import numpy as np

from gensim.models.word2vec import Word2Vec

from utils.config_reader import load_kSVD_config, load_config
from utils.os_utils import grab_files

cargs = load_kSVD_config()
_DEFAULT_MODEL_DIRECTORY  = "models"
F_MODELS = grab_files("*.word2vec",_DEFAULT_MODEL_DIRECTORY)
FLAG_FORCE = cargs.pop("FLAG_FORCE")

def compute_kSVD(item):
    f_model,rng = item

    # Set the random state to be different per sample
    rs = np.random.RandomState(rng)
    
    print "LOADING model", f_model
    model = Word2Vec.load(f_model)
    X = model.syn0

    from ksvd import KSVD

    result = KSVD(X,
              dict_size=cargs["basis_size"],
              target_sparsity=cargs["sparsity"],
              max_iterations=cargs["iterations"],
              enable_printing=True,
              enable_threading = True,
              print_interval=1)

    # Returns tuple (D, Gamma), where X \simeq Gamma * D.
    D,gamma = result
    return f_model,D,gamma


if __name__ == "__main__":

    f_output = "collated/kSVD.h5"

    if os.path.exists(f_output) and not FLAG_FORCE:
        msg = "{} already exists! Add FLAG_FORCE to [kSVD] config to continue"
        raise ValueError(msg.format(f_output))

    # Create a new h5 file first
    h5 = h5py.File(f_output,'w')
    h5.close()

    RNG_COUNTER = itertools.count(42)

    INPUT_ITR = [(f, RNG_COUNTER.next())
                 for f in range(cargs["samples"]) for f in F_MODELS]

    ITR = itertools.imap(compute_kSVD, INPUT_ITR)

    import multiprocessing
    MP = multiprocessing.Pool()
    ITR = MP.imap(compute_kSVD, INPUT_ITR)

    h5 = h5py.File("collated/kSVD.h5",'r+')
    for k,result in enumerate(ITR):
        f_model,D,gamma = result
        print "Saving kSVD {} sample {}".format(f_model, k)

        g_model = h5.require_group(f_model)
        g = g_model.require_group(str(k))

        g.create_dataset("D",data=D, compression="gzip")
        g.create_dataset("gamma",data=gamma, compression="gzip")

        g.attrs["sample_n"] = k
        for key in cargs:
            g.attrs[key] = cargs[key]
