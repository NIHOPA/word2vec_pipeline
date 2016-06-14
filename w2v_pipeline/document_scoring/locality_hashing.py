from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

class RBP_hasher(object):

    def __init__(self, dimension, n_bit):

        self.n_bit = n_bit
        self.dim = dimension

        self.sample_space = 2**n_bit
        
        self.rbp = RandomBinaryProjections('rbp', self.n_bit)
        self.engine = Engine(dimension, lshashes=[self.rbp])

    def _string2int(self,s):
        return int(s,2)

    def __call__(self,v):
        '''
        Convert the returned string into a integer.
        Return a dict based off the weights.
        '''
        s = self.rbp.hash_vector(v)[0]
        weights = {
            self._string2int(s):1.0,
        }

        return weights
