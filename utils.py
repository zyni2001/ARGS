import numpy as np
import scipy.sparse as sp
import torch

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix
    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized
    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.tensor(mx)
    return mx

def torch_normalize_adj(adj):
    adj = adj + torch.eye(adj.shape[0]).cuda()
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).cuda()
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
