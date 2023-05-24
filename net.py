import torch
import torch.nn as nn
import utils

class net_gcn(nn.Module):

    def __init__(self, embedding_dim, adj, features):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.adj_nonzero = torch.nonzero(adj, as_tuple=False).shape[0]

        ## generate the train mask -- this creates a mask = adj.
        self.adj_mask1_train = nn.Parameter(self.generate_adj_mask(adj))
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

