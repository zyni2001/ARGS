import torch
import torch.nn as nn
import copy
import utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances



class net_gcn(nn.Module):

    def __init__(self, embedding_dim, adj, features):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]

        #--------if not use  similarity----------
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))

        #--------if use cosine similarity----------
        ##generate the similarity mask for first iteration training -- this creats a mask = adj * cosine similarity 
        # self.adj_mask1_train = nn.Parameter(self.generate_cos_mask(adj, features))
        ## generate the fixed similarity mask for following iteration training -- this creats a mask = adj * cosine similarity 
        # self.sim_score = self.generate_cos_mask(adj, features)

        #--------if use jaccard similarity----------
        ##generate the similarity mask for first iteration training -- this creats a mask = adj * jaccard similarity 
        # self.adj_mask1_train = nn.Parameter(self.generate_jac_mask(adj, features))
        ## generate the fixed similarity mask for following iteration training -- this creats a mask = adj * jaccard similarity 
        # self.sim_score = self.generate_jac_mask(adj, features)

        ## generate the fixed mask -- this creates a mask = adj.
        self.adj_mask2_fixed = nn.Parameter(self.generate_adj_mask(adj), requires_grad=False)
        self.normalize = utils.torch_normalize_adj    ### A' = D^(-0.5)(A+1)D^(-0.5)

    def forward(self, x, adj, val_test=False):
        
        #save the unchanged features to calculate the smooth feature loss on line 60
        features = x

        adj = torch.mul(adj, self.adj_mask1_train)  ### mul -- element by element multiplication.
        adj = torch.mul(adj, self.adj_mask2_fixed)
        adj = self.normalize(adj)
        #adj = torch.mul(adj, self.adj_mask2_fixed)
        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        

        # calculate the smooth feature loss
        loss_smooth_feat = self.feature_smoothing(adj, features) #* loss_coefficient # 0.01 is to make loss_feat reach the same scale of CE loss
        
    
        return x, loss_smooth_feat

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)   #### the mask is 1 and 0--- 1 means edge is there. 0 means edge is not there.
        return mask
    
    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

    def generate_cos_mask(self, input_adj, features):
        #### get features and adjancency matrix ####
        fea_copy = features.cpu().data.numpy()
        mask_fixed = input_adj.cpu().data.numpy()

        # try cosine similarity matrix
        sim_matrix = cosine_similarity(fea_copy)  

        if mask_fixed.shape != sim_matrix.shape:
            print("The tensors have different shapes.")

        # mask = similarity matrix * adj
        mask_train = sim_matrix * mask_fixed
        mask_train = torch.tensor(mask_train).cuda()

        #----------L1 normalize----------
        # mask_train = normalize(mask_train, axis=1, norm='l1')

        #----------softmax normalize of each row ----------
        # sim_matrix = torch.softmax(sim_matrix, dim=1)

        #----------softmax normalize on all non-zero values----------
        # non_zero_indices = torch.nonzero(mask_train)
        # mask_train[non_zero_indices[:, 0], non_zero_indices[:, 1]] = torch.softmax(mask_train[non_zero_indices[:, 0], non_zero_indices[:, 1]], dim=0)

        return mask_train.float()

    def generate_jac_mask(self, input_adj, features):   
        # Convert the PyTorch tensor to a NumPy array
        features_np = features.cpu().data.numpy()

        # Calculate the Jaccard similarity score matrix
        jaccard_matrix = pairwise_distances(features_np, metric='jaccard')

        # 1-matrix
        jaccard_matrix = 1 - jaccard_matrix

        # Convert the result back to a PyTorch tensor if necessary
        jaccard_matrix = torch.tensor(jaccard_matrix).cuda()

        # multiply with adj matrix
        jaccard_matrix = torch.mul(input_adj, jaccard_matrix)

        #----------softmax normalize on all non-zero values----------
        # non_zero_indices = torch.nonzero(jaccard_matrix)
        # jaccard_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]] = torch.softmax(jaccard_matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]], dim=0)

        return jaccard_matrix.float()

class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

class net_gcn_admm(nn.Module):

    def __init__(self, embedding_dim, adj):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]
        self.adj_layer1 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        self.adj_layer2 = nn.Parameter(copy.deepcopy(adj), requires_grad=True)
        
    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            if ln == 0:
                x = torch.mm(self.adj_layer1, x)
            elif ln == 1:
                x = torch.mm(self.adj_layer2, x)
            else:
                assert False
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x

    # def forward(self, x, adj, val_test=False):

    #     for ln in range(self.layer_num):
    #         x = torch.mm(self.adj_list[ln], x)
    #         x = self.net_layer[ln](x)
    #         if ln == self.layer_num - 1:
    #             break
    #         x = self.relu(x)
    #         if val_test:
    #             continue
    #         x = self.dropout(x)
    #     return x

    def generate_adj_mask(self, input_adj):
        
        sparse_adj = input_adj
        zeros = torch.zeros_like(sparse_adj)
        ones = torch.ones_like(sparse_adj)
        mask = torch.where(sparse_adj != 0, ones, zeros)
        return mask

class net_gcn_baseline(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj, val_test=False):

        for ln in range(self.layer_num):
            x = torch.mm(adj, x)
            # x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)
        return x


class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss

