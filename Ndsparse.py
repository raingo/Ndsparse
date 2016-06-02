from collections import defaultdict
import operator
from functools import wraps
import numpy as np

SUPPORTED_DTYPE = (int, long, float, complex)
def pairwise(fn):
    @wraps(fn)
    def wrapper(self, other):
        # if other is scalar, only create a new one with the same nnz positions
        # so the add and sub are different from regular
        # !!! support broadcast
        if isinstance(other, SUPPORTED_DTYPE):
            keys = self.entries.keys()
            entries = {key:other for key in keys}
            other = self.__class__(entries)
        assert len(self.shape) == len(other.shape), "shape mismatch"
        lhs = self.repmat(other.shape)
        rhs = other
        if lhs is None:
            lhs = other.repmat(self.shape)
            rhs = self
            assert lhs is not None, \
                    "do not know how to broadcast: %s vs %s" \
                    % (self.shape, other.shape)

        return fn(lhs, rhs)
    return wrapper

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
        NDsparse(nested list dense representation)
        """
        # NDsparse from a single scalar
        if isinstance(args[0], SUPPORTED_DTYPE):
            self.entries = {():args[0]}
            self.shape = ()
            self.dtype = type(args[0])

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

        # NDsparse from list of lists (of lists...) dense format
        # 1st dim = rows, 2nd dim = cols, 3rd dim = pages, ...
        elif args[0].__class__.__name__ == 'list':
            self.entries = buildEntriesDictFromNestedLists(args[0])
            self.shape = getListsShape(args[0])

        # Catch unsupported initialization
        else:
            raise Exception("Improper Ndsparse construction.")

        try:
            key, value = self.entries.iteritems().next()
        except:
            value = 1.
        self.dtype = type(value)

        # Cleanup
        self.removeZeros()

    @property
    def ndim(self):
        return len(self.shape)

    def repmat(self, newshape):
        diff_index = [(idx, s0, s1) for idx, (s0, s1)
                in enumerate(zip(self.shape, newshape))
                if s0 != s1]
        if any([f[1] != 1 for f in diff_index]):
            return None

        prev = self.entries
        for idx, _, s1 in diff_index:
            res = {}
            for key, value in prev.items():
                key = list(key)
                for i in range(s1):
                    key[idx] = i
                    res[tuple(key)] = value
            prev = res
        return self.__class__(prev)

    def copy(self):
        """
        Copy "constructor"
        """
        return Ndsparse(self.entries)

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    def swapaxes(self, A, B):
        if A == B:
            return

        idxs = range(self.ndim)
        idxs[A], idxs[B] = idxs[B], idxs[A]
        self.transpose(idxs)

    def ravel(self):
        raise NotImplemented

    def _reduce(self, axis, op):
        full = range(self.ndim)
        if axis is None:
            axis = full

        keep = sorted(list(set(full) - set(axis)))
        def _sub(key):
            return tuple([key[i] for i in keep])

        mapping = defaultdict(list)
        for key, value in self.entries.iteritems():
            mapping[_sub(key)].append(value)

        res = {}
        for key, values in mapping.iteritems():
            res[key] = op(values)

        if len(keep) == 0:
            return res[()]
        else:
            return self.__class__(res)

    def sum(self, axis=None):
        return self._reduce(axis=axis,op=sum)

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
                if value is not None]
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
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] + other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = other.entries[pos]

        return Ndsparse(out,self.shape)

    @pairwise
    def __sub__(self,other):
        """
        Elementwise subtraction of self - other.
        """
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] - other.entries[pos]
        for pos in selfFree:
            out[pos] = self.entries[pos]
        for pos in otherFree:
            out[pos] = -other.entries[pos]

        return Ndsparse(out,self.shape)

    @pairwise
    def __mul__(self,other):
        """
        Elementwise multiplication of self .* other.
        """
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]

        return Ndsparse(out,self.shape)

    @pairwise
    def __div__(self,other):
        """
        Elementwise division of nonzero entries of self ./ other, casting ints to floats.
        """
        overlap, selfFree, otherFree = self.mergePositions(other)
        out = {}

        for pos in overlap:
            out[pos] = float(self.entries[pos]) / other.entries[pos]

        return Ndsparse(out,self.shape)

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

    def reshape(self, shapemat):
        """
        Like the MATLAB reshape. http://www.mathworks.com/help/matlab/ref/reshape.html
        """
        assert reduce(operator.mul, shapemat, 1) == self.size, \
                'reshape can not change the number of elements'
        def _map(key):
            index = np.ravel_multi_index(key, self.shape)
            return np.unravel_index(index, shapemat)

        res = {}
        for key, value in self.entries.items():
            res[_map(key)] = value

        self.entries = res
        self.shape = tuple(shapemat)

def permute(vec,permutation):
    """
    Permute vec tuple according to permutation tuple.
    """
    return tuple([vec[permutation[i]] for i in range(len(vec))])

def traverseWithIndices(superList, treeTypes=(list, tuple)):
    """
    Traverse over tree structure (nested lists), with indices. Returns a nested list
    with the left element a nested list and the right element the next position.
    Call flatten to get into a nice form.
    """
    idxs = []
    if isinstance(superList, treeTypes):
        for idx,value in enumerate(superList):
            idxs = idxs[0:-1]
            idxs.append(idx)
            for subValue in traverseWithIndices(value):
                yield [subValue,idxs]
    else:
        yield [superList,idxs]

def flatten(superList, treeTypes=(list, tuple)):
    '''
    Flatten arbitrarily nested lists into a single list, removing empty lists.
    '''
    flatList = []
    for subList in superList:
        if isinstance(subList, treeTypes):
            flatList.extend(flatten(subList))
        else:
            flatList.append(subList)
    return flatList

def buildEntriesDictFromNestedLists(nestedLists):
    """
    Build dict of pos:val pairs for Ndsparse.entries format from a flat list where list[0] is
    the val and list[1:-1] are the pos indices in reverse order.
    """
    # Special case for scalar in a list
    #    Warning: first dim shouldn't be singleton
    if len(nestedLists) == 1:
        return {():nestedLists[0]}

    entriesDict = {}
    for entry in traverseWithIndices(nestedLists):
        flatEntry = flatten(entry)
        pos = tuple(flatEntry[-1:0:-1])
        val = flatEntry[0]
        entriesDict[pos] = val
    return entriesDict

def getListsShape(nestedLists, treeTypes=(list, tuple)):
    """
    Get dimensions of nested lists
    """
    shape = []
    lst = list(nestedLists)
    while isinstance(lst, treeTypes):
        shape.append(len(lst))
        lst = lst[0]
    return tuple(shape)

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
