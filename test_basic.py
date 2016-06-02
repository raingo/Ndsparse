#!/usr/bin/env python
"""
Code description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = "Raingo Lee (raingomm@gmail.com)"

import sys
import os.path as osp

from Ndsparse import Ndsparse

def main():
  data = {(0,0,0):1, (20,10,100):1}
  mat = Ndsparse(data)
  print(mat.dtype)
  print(mat.size)
  print(mat.ndim)

  mat.swapaxes(0,2)
  print(mat.shape)
  mat.swapaxes(0,2)
  print(mat.shape)

  print(mat.sum())
  print(mat.sum(axis=(1,2)))

  print(mat.max())
  print(mat.max(axis=(1,2)))

  print(mat[3,4,2])

  print(mat * 5)

  # test broadcast
  print(mat.shape)
  mat2 = mat[None, None, None, None]
  print(mat2.shape)
  mat3 = mat2.repmat(mat.shape+(3,))
  print(mat2.shape)
  print(mat3.shape)
  print(mat2 + mat3)
  # test pairwise
  # test slicing

  print(mat)
  mat.reshape(mat.shape+(1,1,1))
  print(mat)

  pass

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
