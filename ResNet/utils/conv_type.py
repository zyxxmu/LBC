import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from itertools import combinations
from utils.options import args
LearnedBatchNorm = nn.BatchNorm2d
N = args.N
M = args.M

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class NMConv(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N  # number of non-zeros
        self.M = M
        self.mask_shape = self.weight.clone().permute(0, 2, 3, 1).shape
        self.group = int(self.weight.numel() / self.M)
        self.alpha, self.index_list = self.init(self.N, self.M)  

    def init(self, N, M):
        list1 = list(range(1, M + 1))
        list2 = list(combinations(list1, N))
        list3 = list(range(1, M))
        list4 = list(combinations(list3, N-1))
        size = len(list2)
        size_2 = len(list4)
        index_list = torch.zeros(size, M)
        for i in range(size):
            for j in range(N):
                index_list[i][list2[i][j] - 1] = 1
        return nn.Parameter(torch.full([self.group, size], 1 / size_2)), nn.Parameter(index_list, requires_grad=False)

    def forward(self, x):
        index = (self.alpha != 0).type_as(self.alpha)
        alpha = self.alpha * index
        mask = (alpha @ self.index_list).reshape(self.mask_shape)
        mask = GetMask.apply(mask)
        sparseWeight = mask.permute(0,3,1,2) * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        index = mask == 0
        mask2 = mask + index
        mask = mask / mask2.detach()
        return mask

    @staticmethod
    def backward(ctx, g):
        return g
