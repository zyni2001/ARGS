import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class AddTrainableMask(ABC):

    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):

        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        
        return pruned_weight

    @classmethod   ### here you remving the original weight parameter and adding mask_train, mask_fixed, and orig_weight parameters.
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):
        #### module here is a nn layer. 
        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)
        print("orig is:", orig.dtype)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method

def add_mask(model, init_mask_dict=None):

    if init_mask_dict is None:
        
        mask1_train = nn.Parameter(torch.ones_like(model.net_layer[0].weight))
        mask1_fixed = nn.Parameter(torch.ones_like(model.net_layer[0].weight), requires_grad=False)
        mask2_train = nn.Parameter(torch.ones_like(model.net_layer[1].weight))
        mask2_fixed = nn.Parameter(torch.ones_like(model.net_layer[1].weight), requires_grad=False)
        
    else:
        mask1_train = nn.Parameter(init_mask_dict['mask1_train'])
        mask1_fixed = nn.Parameter(init_mask_dict['mask1_fixed'], requires_grad=False)
        mask2_train = nn.Parameter(init_mask_dict['mask2_train'])
        mask2_fixed = nn.Parameter(init_mask_dict['mask2_fixed'], requires_grad=False)

    AddTrainableMask.apply(model.net_layer[0], 'weight', mask1_train, mask1_fixed)
    AddTrainableMask.apply(model.net_layer[1], 'weight', mask2_train, mask2_fixed)

def print_sparsity(model):

    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight1_total = model.net_layer[0].weight_mask_fixed.numel()
    weight2_total = model.net_layer[1].weight_mask_fixed.numel()
    weight_total = weight1_total + weight2_total

    weight1_nonzero = model.net_layer[0].weight_mask_fixed.sum().item()
    weight2_nonzero = model.net_layer[1].weight_mask_fixed.sum().item()
    weight_nonzero = weight1_nonzero + weight2_nonzero

    wei_spar = weight_nonzero * 100 / weight_total
    # print("-" * 100)
    # print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
    # .format(adj_spar, wei_spar))
    # print("-" * 100)

    return adj_spar, wei_spar
