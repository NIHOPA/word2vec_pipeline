from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections


class RBP_hasher(object):

    def __init__(self, dimension, n_bit, alpha):

        self.n_bit = n_bit
        self.dim = dimension
        self.alpha = alpha

        self.sample_space = 2**n_bit

        self.rbp = RandomBinaryProjections('rbp', self.n_bit)
        self.engine = Engine(dimension, lshashes=[self.rbp])

    @property
    def params(self):
        return self.rbp.get_config()

    def load(self, config):
        self.rbp.apply_config(config)

    def _string2int(self, s):
        return int(s, 2)

    def __call__(self, v):
        '''
        Convert the returned string into a integer.
        Return a dict based off the weights.
        '''
        s = self.rbp.hash_vector(v)[0]
        weights = {
            self._string2int(s): 1.0,
        }

        if not self.alpha:
            return weights

        # If alpha is non-zero, deposit weight into nearby bins

        slist = map(bool, map(int, list(s)))
        for n in range(len(s)):
            s2list = slist[:]
            s2list[n] = not slist[n]
            s2list = map(str, map(int, s2list))
            s2 = ''.join(s2list)
            idx = self._string2int(s2)
            weights[idx] = self.alpha

        return weights
