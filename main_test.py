import argparse
import torch
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
import dgl

import net as net
import pruning
import utils

def test(args, seed):
    # set up seed
    pruning.setup_seed(seed)

    # load modified graph
    from pathlib import Path
    dataset_path = Path("./mod_graph/")
    output_graph = Path(dataset_path).joinpath(
        "ptb_graph",
        args["attack_name"],
        args["datasource"],
        f"ptb_rate_"+str(args["ptb_rate"]),
        args["dataset"],
    )
    # read the record to find the best seed
    with open(output_graph.joinpath("archive.txt"), 'r') as f:
        best_seed = f.readline().strip().split()[0]
    # set up the path with the best seed
    output_graph = output_graph.joinpath(f"seed_{best_seed}")
    print("Loading data from " + str(output_graph) + " ......")
    mod_data = dgl.data.CSVDataset(output_graph)
    mod_g = mod_data[0]

    # load features and adjacency matrix
    #### get adj ####
    adj = mod_g.adjacency_matrix().to_dense()
    #### get train, val, and test sets ####
    idx_val = torch.nonzero(mod_g.ndata['val_mask']).squeeze().tolist()
    idx_test = torch.nonzero(mod_g.ndata['test_mask']).squeeze().tolist()
    #### get features ####
    features_temp = mod_g.ndata["feat"]
    features = utils.normalize_feature(features_temp)
    #### get labels ####
    labels = mod_g.ndata["label"]
    # load cuda
    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    # define model
    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj, features=features)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()

    # load the model parameters and graph adjacency matrix
    state_dict = torch.load(args['model_path'])
    net_gcn.load_state_dict(state_dict)

    ### print test results
    # adj_spar - means how many edges in graph are present.
    # wei_spar - means how many edges in model are present.
    adj_spar, wei_spar = pruning.print_sparsity(net_gcn) 
    print("-" * 100)
    print("Graph Sparsity: [{:.2f}%] Model Sparsity:[{:.2f}%]"
    .format(100-adj_spar, 100-wei_spar))
    print("-" * 100)
    with torch.no_grad():
        output, loss_feat_ = net_gcn(features, adj, val_test=True)
        acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
        acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')  
    print("Val:[{:.2f}] Test:[{:.2f}] "
        .format(acc_val * 100, acc_test * 100))
    print("-" * 100)

def parser_loader():
    parser = argparse.ArgumentParser(description='ARGS')
    ###### main settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--attack_name', type=str, default='pgd_meta_adam', choices=['pgd_meta_adam', 'pgd', 'mettack'], help='attack types')
    parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
    parser.add_argument('--model_path', type=str, default='gcn_cora_pgd_ptb_rate_0.05_best_model.pth', help='best model path')
    # default setting
    parser.add_argument('--datasource', type=str, default='deeprobust_nettack_orilabel', choices=['planetoid_fakelabel', 'planetoid_orilabel', 'deeprobust_nettack_orilabel'], help='datasource')
    return parser

if __name__ == "__main__":    
    parser = parser_loader()
    args = vars(parser.parse_args())
    seed_dict = {'cora': 2377, 'citeseer': 4428, 'pubmed': 3333} 
    seed = seed_dict[args['dataset']]

    # Construct the model path based on the inputs
    args['model_path'] = f'model/gcn_{args["dataset"]}_{args["attack_name"]}_ptb_rate_{args["ptb_rate"]}_best_model.pth'
    
    test(args, seed) 

