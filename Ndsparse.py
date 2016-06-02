from collections import defaultdict
import operator
from functools import wraps
import numpy as np

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
log_space = True

import math
def logsumexp(target):
    max_v = max(target)
    return max_v + math.log(sum([math.exp(t-max_v) for t in target]))

SUPPORTED_DTYPE = (int, long, float, complex)
def pairwise(fn):
    only_overlap = fn.func_name in ['__div__', '__mul__']
    @wraps(fn)
    def wrapper(self, other):
        # if other is scalar, only create a new one with the same nnz positions
        # so the add and sub are different from regular
        # !!! support broadcast
        if isinstance(other, SUPPORTED_DTYPE):
            keys = self.entries.keys()
            entries = {key:other for key in keys}
            other = self.__class__(entries, self.shape)
        assert len(self.shape) == len(other.shape), "shape mismatch"

        if only_overlap:
            lhs, rhs = overlap(self, other)
        else:
            # in the case of (2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,) and (2,)
            # repmat is very inefficient
            lhs = self.repmat(other.shape)
            rhs = other.repmat(lhs.shape)

        assert lhs.shape == rhs.shape, \
              "do not know how to broadcast: %s vs %s" \
              % (self.shape, other.shape)

        return fn(lhs, rhs)
    return wrapper

def overlap(mat0, mat1):
    def _collapse(keys, keeps):
        res = defaultdict(list)
        for key in keys:
            res[tuple([key[i] for i in keeps])].append(key)
        return res

    shared_dims = [idx for idx, (s0, s1)
                in enumerate(zip(mat0.shape, mat1.shape))
                if s0 == s1]

    primary1 = [idx for idx, (s0, s1)
                in enumerate(zip(mat0.shape, mat1.shape))
                if s0 != s1 and s0 == 1]

    mA = _collapse(mat0.entries.keys(), shared_dims)
    mB = _collapse(mat1.entries.keys(), shared_dims)

    kO = set(set(mA.keys()) & set(mB.keys()))

    # 3,4,1 x 3,1,5
    # 3,4,5
    shape = list(mat0.shape)
    for p in primary1:
        shape[p] = mat1.shape[p]

    A, B= {}, {}
    for k in kO:
        for kA in mA[k]:
            for kB in mB[k]:
                key = list(kA)
                # kA: 1,3,1
                # kB: 1,1,4
                # key: 1,3,4
                for p in primary1:
                    key[p] = kB[p]
                A[tuple(key)] = mat0.entries[kA]
                B[tuple(key)] = mat1.entries[kB]
    return Ndsparse(A, shape), Ndsparse(B, shape)

class Ndsparse:
    """
    N-dimensional sparse matrix.
    entries: dict of positions and values in matrix
        key: N-tuple of positions (i,k,...)
        val: value at position
    ndim: dimension
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
        NDsparse(scalar)
        NDsparse(dict of (pos):val pairs, optional list of dims for shape)
        """
        # NDsparse from a single scalar
        if isinstance(args[0], SUPPORTED_DTYPE):
            self.entries = {():args[0]}
            self.shape = ()

        # NDsparse from dict of pos,val pairs
        elif args[0].__class__.__name__ == 'dict':
            # Error handling:
            # Make sure all keys in dict are same length
            # Make sure all indexes in keys are ints
            # Make sure all vals in dict are numbers
            self.entries = args[0]
            if len(args) > 1:
                self.shape = tuple(args[1])
            else:
                self.shape = getEntriesShape(args[0])

        # Catch unsupported initialization
        else:
            raise Exception("Improper Ndsparse construction.")

        try:
            key, value = self.entries.iteritems().next()
        except:
            value = 1.
        self.dtype = float

        # Cleanup
        if not log_space:
            self.removeZeros()

    @property
    def ndim(self):
        return len(self.shape)

    def repmat(self, newshape):
        diff_index = [(idx, s0, s1) for idx, (s0, s1)
                in enumerate(zip(self.shape, newshape))
                if s0 != s1 and s0 == 1]

        shape = list(self.shape)
        for idx, _, _ in diff_index:
            shape[idx] = newshape[idx]

        prev = self.entries
        for idx, _, s1 in diff_index:
            res = {}
            for key, value in prev.items():
                key = list(key)
                for i in range(s1):
                    key[idx] = i
                    res[tuple(key)] = value
            prev = res
        return self.__class__(prev, shape)


    def is_close(self, other):
        if self.shape != other.shape:
            return False

        kA = sorted(self.entries.keys())
        kB = sorted(other.entries.keys())

        if kA != kB:
            return False

        for k in kA:
            if not isclose(self.entries[k], other.entries[k]):
                return False
        return True

    def copy(self):
        """
        Copy "constructor"
        """
        new = {}
        for k, v in self.entries.items():
            new[k] = v
        return self.__class__(new, self.shape)

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    def swapaxes(self, A, B):
        if A == B:
            return self

        idxs = range(self.ndim)
        idxs[A], idxs[B] = idxs[B], idxs[A]
        return self.transpose(idxs)

    def ravel(self):
        raise NotImplemented

    def _reduce(self, axis, op):
        full = range(self.ndim)
        if axis is None:
            axis = full

        keep = sorted(list(set(full) - set(axis)))
        def _sub(key):
            return tuple([key[i] for i in keep])

        new_shape = tuple([self.shape[i] for i in keep])

        mapping = defaultdict(list)
        for key, value in self.entries.iteritems():
            mapping[_sub(key)].append(value)

        res = {}
        for key, values in mapping.iteritems():
            res[key] = op(values)

        if len(keep) == 0:
            if () in res:
                return res[()]
            else:
                return 0.
        else:
            return self.__class__(res, new_shape)

    def sum(self, axis=None, dtype=None, out=None):
        # to compatible with np.sum()
        # but the dtype and out are ignored
        if log_space:
            op = logsumexp
        else:
            op = sum
        return self._reduce(axis=axis,op=op)

    def max(self, axis=None):
        return self._reduce(axis=axis,op=max)

    def __getitem__(self, key):
        # if key[i] == None, all elements along that direction is returned
        if len(key) > self.ndim:
            # all key are None, equal to reshape
            assert all([k is None for k in key[self.ndim:]]), \
                    "invalid index: %s for ndim: %d" % (key, self.ndim)
            self = self.copy()
            self.reshape(self.shape+(1,)*(len(key)-self.ndim))

        slices = [(idx, value) for idx, value in enumerate(key)
                if value is not None and not isinstance(value, slice)]
        if len(slices) == 0:
            return self

        assert all([self.shape[k] > v for k, v in slices]), \
                'invalid index: %s for shape: %s' % (key, self.shape)
        keep = sorted(list(set(range(self.ndim)) - set([f[0] for f in slices])))

        def _map(key):
            return tuple([key[f] for f in keep])

        res = {}
        new_shape = [self.shape[k] for k in keep]
        for key, value in self.entries.items():
            if all([key[k] == v for k, v in slices]):
                # got one
                res[_map(key)] = value
        return self.__class__(res, new_shape)

    def __repr__(self):
        """
        String representation of Ndsparse class
        """
        rep = []
        rep.append(''.join([str(self.ndim),'-d sparse tensor with ', str(self.nnz()), ' nonzero entries\n']))
        poss = self.entries.keys()
        poss.sort()
        for pos in poss:
            rep.append(''.join([str(pos),'\t',str(self.entries[pos]),'\n']))
        return ''.join(rep)

    def nnz(self):
        """
        Number of nonzero entries. Number of indexed entries if no explicit 0's allowed.
        """
        return len(self.entries)

    def addEntry(self,pos,val):
        # Error handling: make sure entry doesn't overwrite existing ones
        self.entries[pos] = val

    def addEntries(self,newEntries):
        # Error handling: make sure entries don't overwrite existing ones
        self.entries.update(newEntries)

    def mergePositions(self,other):
        """
        Return (overlap, selfFree, otherFree)
            overlap: set of tuples of positions where self and other overlap
            selfFree: set of tuples of positions where only self is nonzero
            otherFree: set of tuples of positions where only other is nonzero
        """
        selfKeys = set(self.entries.keys())
        otherKeys = set(other.entries.keys())
        overlap = selfKeys & otherKeys
        selfFree = selfKeys.difference(otherKeys)
        otherFree = otherKeys.difference(selfKeys)
        return (overlap, selfFree, otherFree)

    def removeZeros(self):
        """
        Remove explicit 0 entries in Ndsparse matrix
        """
        newEntries = {}
        for pos,val in self.entries.iteritems():
            if val != 0:
                newEntries[pos] = val
        self.entries = newEntries

    def __eq__(self,other):
        """
        Test equality of 2 Ndsparse objects. Must have the same nonzero elements, rank, and dimensions.
        """
        if self.ndim == other.ndim and self.shape == other.shape and self.entries == other.entries:
            return True
        else:
            return False

    @pairwise
    def __add__(self,other):
        """
        Elementwise addition of self + other.
        """
        assert not log_space, "log space do not support add"
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] + other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = other.entries[pos]

        return self.__class__(out, self.shape)

    @pairwise
    def __sub__(self,other):
        """
        Elementwise subtraction of self - other.
        """
        assert not log_space, "log space do not support sub"
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] - other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = -other.entries[pos]

        return self.__class__(out, self.shape)

    @pairwise
    def __mul__(self,other):
        """
        Elementwise multiplication of self .* other.
        """
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            if log_space:
                out[pos] = self.entries[pos] + other.entries[pos]
            else:
                out[pos] = self.entries[pos] * other.entries[pos]

        return self.__class__(out,self.shape)

    @pairwise
    def __div__(self,other):
        """
        Elementwise division of nonzero entries of self ./ other, casting ints to floats.
        """
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            if log_space:
                out[pos] = float(self.entries[pos]) - other.entries[pos]
            else:
                out[pos] = float(self.entries[pos]) / other.entries[pos]

        return self.__class__(out,self.shape)

    __truediv__ = __div__

    def transpose(self,permutation):
        """
        Transpose Ndsparse matrix in place
        permutation: tuple of new indices
        Matrix starts out as (0,1,...,N) and can be transposed according to
           the permutation (N,1,....0) or whatever, with N! possible permutations
        Note indexing starts at 0
        """
        # Error handling: make sure permutation is valid (eg, has right length)
        # Useful extension: default transpose for N=2 matrices
        out = {}
        for key,value in self.entries.iteritems():
            out[permute(key,permutation)] = value
        self.entries = out
        self.shape = tuple(permute(self.shape,permutation))
        return self

    def to_numpy(self):
        res = np.zeros(self.shape)
        for key, value in self.entries.items():
            if log_space:
                res[key] = math.exp(value)
            else:
                res[key] = value
        return res

    def reshape(self, shapemat):
        """
        Like the MATLAB reshape. http://www.mathworks.com/help/matlab/ref/reshape.html
        """
        assert reduce(operator.mul, shapemat, 1) == self.size, \
                'reshape can not change the number of elements'
        shapemat = tuple(shapemat)

        def _map0(key):
            # FIXME: the following two functions fail with ndim > 32
            index = np.ravel_multi_index(key, self.shape)
            return np.unravel_index(index, shapemat)

        def isexpand():
            # (2, 3, 4)
            # (2, 3, 4, 1, 1)
            return len(shapemat) > len(self.shape) \
                    and shapemat[:len(self.shape)] == self.shape \
                    and all([s==1 for s in shapemat[(len(self.shape)+1):]])

        def _map1(key):
            key += (0,) * (len(shapemat) - len(self.shape))
            return key

        if self.shape == shapemat:
            _map = lambda x:x
        elif isexpand():
            _map = _map1
            pass
        else:
            _map = _map0

        res = {}
        for key, value in self.entries.items():
            res[_map(key)] = value

        self.entries = res
        self.shape = tuple(shapemat)
        return self

def permute(vec,permutation):
    """
    Permute vec tuple according to permutation tuple.
    """
    return tuple([vec[permutation[i]] for i in range(len(vec))])

def getEntriesShape(entries):
    """
    Get dimensions corresponding to max indices in entries
    """
    maxIdxs = [0]*len(entries.iterkeys().next())
    for pos in entries.iterkeys():
        for i,idx in enumerate(pos):
            if idx > maxIdxs[i]:
                maxIdxs[i] = idx
    return tuple([idx+1 for idx in maxIdxs])
