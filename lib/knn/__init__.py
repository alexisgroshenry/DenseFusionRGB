import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from lib.knn import knn_pytorch as knn_pytorch

class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  @staticmethod
  def forward(self, k, ref, query):
    self.k = k
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda()

    knn_pytorch.knn(ref, query, inds)

    return inds


class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    while(1):
        D, N, M = 128, 100, 1000
        ref = Variable(torch.rand(2, D, N))
        query = Variable(torch.rand(2, D, M))

        inds = KNearestNeighbor.apply(2, ref, query)
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
        #ref = ref.cpu()
        #query = query.cpu()
        print(inds)


if __name__ == '__main__':
  unittest.main()
