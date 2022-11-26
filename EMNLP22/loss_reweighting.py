# coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utilis.sqrtm import sqrtm


def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    # vals, vecs = torch.linalg.eig(matrix)
    vals = torch.view_as_complex(vals.contiguous())+0.001
    vals_pow = vals.pow(p)
    vals_pow = torch.view_as_real(vals_pow)[:, 0]
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow


def lossb_expect(cfeaturec, weight):
    feature_m = F.normalize(cfeaturec, p=2, dim=1).cuda()    # feature+pre_feature(64) * 768
    similar_m = torch.matmul(feature_m, feature_m.t())   # 64 * 768  X  768 * 64 -->   64 * 64
    # cfeaturec = cfeaturec.cuda()
    # similar_m = torch.cosine_similarity(cfeaturec.unsqueeze(1), cfeaturec.unsqueeze(0), dim=2).cuda()

    sqt_m = _matrix_pow(similar_m, -0.5)
    # todo 伪逆
    nys_m = torch.matmul(similar_m, sqt_m)
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()

    cov1 = cov(nys_m, weight)
    cov_matrix = cov1 * cov1
    loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    print(loss)
    return loss


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)  # 32*2 -> 64
        cov = torch.matmul((w * x).t(), x)   # x： [这次的feature 之前的feature一起过一个RFF  , 768]   w: [这次的weight（全是一个数字） 上次的weight]
        # 768 * 64 matmul 64 * 768 -->  768 * 768   x： 64 * 768 这个时候已经当他768的每一维度都是独立的特征采样结果了
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res


# cfeaturec = torch.rand([64, 768]).cuda()
# weight = F.normalize(torch.ones([64, 1]), dim=0).cuda()
# res = lossb_expect(cfeaturec, weight, 1)
# print(res)

if __name__ == '__main__':
    pass
