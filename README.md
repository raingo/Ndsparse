Ndsparse
========

N-dimensional sparse matrix implementation

Supported operations
=======
1. basic slicing. only support `A[:, 3, :]` not `A[2:3, 2]`, and `A[3:4, 3]` will be interpreted as `A[:, 3]`
2. reshape
3. pairwise operations, add, mul, sub, div, with support of broadcast and work with scalars.
4. reduce operations, max and sum, along specified axises, like `numpy.sum()` and `numpy.max()`
5. transpose `numpy.transpose()`
6. swapaxes `numpy.swapaxes()`
7. Efficient broadcast with pairwise multiplication and division, so it's possible to multiply between
```
  (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
```
and
```(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)```
