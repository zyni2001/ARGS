from copy import deepcopy
# from deepgraph.util import *
import torch
import numpy as np


def evaluate(model, adj, feats, labels, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    def get_evaluator():
        def evaluator(out, labels):
            pred = out.argmax(1)
            return pred.eq(labels).float().mean().item()
        return evaluator

    criterion = torch.nn.NLLLoss()
    evaluator = get_evaluator()

    model.eval()
    with torch.no_grad():
        logits = model.forward(adj, feats)
        out = logits.log_softmax(dim=1)
        if idx_eval is None:
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else:
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    return score


def train(model, epochs, optim, adj, run, features, labels, idx_train, idx_val, idx_test, loss, verbose=True):
    best_loss_val = 9999  ### here labels is the pseudo-lables given from MLP
    best_acc_val = 0
    weights = deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        logits = model(adj, features)
        l = loss(logits[idx_train], labels[idx_train])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, features, labels, idx_val)
        val_loss = loss(logits[idx_val], labels[idx_val])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, adj, features, labels, idx_test)
    print("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc


def train_MLP(model, epochs, optimizer, train_loader, val_loader, test_loader, loss, verbose=True):
    model.train()
    best_acc = 0
    best_loss =9999
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            optimizer.zero_grad()
            l = loss(output, y)
            l.backward()
            optimizer.step()
        n_acc = 0
        loss_total = 0
        n = 0
        best_acc_val = 0
        best_loss_val = 0
        model.eval()
        for x, y in val_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            pred = torch.argmax(output, dim=1)
            n += len(y)
            acc = (pred == y).sum().item()
            n_acc += acc
            l = loss(output, y)
            loss_total += l
        acc_total = n_acc / n
        val_loss = loss_total /n
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc_total > best_acc_val:
            best_acc_val = acc_total
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc_total))
    model.load_state_dict(weights)
    model.eval()
    n_acc = 0
    n = 0
    for x, y in test_loader:
        x = x.cuda()
        y = y.cuda()
        output = model(x)
        pred = torch.argmax(output, dim=1)
        n += len(y)
        acc = (pred == y).sum().item()
        n_acc += acc
    return  n_acc / n


def adj_norm(adj, neighbor_only=False):
    if not neighbor_only:
        adj = torch.add(torch.eye(adj.shape[0]).cuda(), adj)
    if adj.is_sparse:
        degree = adj.to_dense().sum(dim=1)
    else:
        degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), -0.5).expand(adj.shape[0], adj.shape[0])
    in_degree_norm = torch.where(torch.isinf(in_degree_norm), torch.full_like(in_degree_norm, 0), in_degree_norm)
    out_degree_norm = torch.pow(degree.view(-1, 1), -0.5).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.where(torch.isinf(out_degree_norm), torch.full_like(out_degree_norm, 0), out_degree_norm)
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    return adj


def get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=30, append_idx=True):
    # idx_train = np.array([], dtype='int32')
    ### logits - mlp prediction; psedu_labels = true label
    if append_idx:
        idx_train = idx_train
    else:
        idx_train = np.array([])
    # pred_labels is the prediction of MLP
    # logits = torch.tensor([
                        #   [0.4, 0.6], # test
                        #   [0.5, 0.5], # train
                        #   [0.9, 0.1], # train
                        #   [0.5, 0.5], # train
                        #   [0.6, 0.4], # test
                        #   [0.4, 0.6], # test
                        #   [0.3, 0.7], # train
                        #   [1.0, 0.0], # test
                        #   [0.5, 0.5], # test
                        #   [0.8, 0.2]  # test
                        # ])
    # idx_test = [0, 4, 5, 7, 8, 9]
    pred_labels = torch.argmax(logits, dim=1) ### predicted MLP label, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    pred_labels_test = pred_labels[idx_test]  ### predicted MLP label for test nodes only, [0, 1, 1, 2, 2, 2]
    for label in range(pseudo_labels.max().item() + 1):
        idx_label = idx_test[pred_labels_test==label] ### for the given label; idx_label is the set of nodes of a given class, [4, 5], label = 1
        logits_label = logits[idx_label][:, label] ## only the logit value for getting top k, tensor([0.6, 0.6])
        if len(logits_label) > k:
            _, idx_topk = torch.topk(logits_label, k)
        else:
            idx_topk = np.arange(len(logits_label)) # [0,1]
        idx_topk = idx_label[idx_topk] ### idx_topk is the top k nodes for the given label, [4, 5]
        pseudo_labels[idx_topk] = label ### assign the pseudo label to original true label
        idx_train = np.concatenate((idx_train, idx_topk)) ### append the top k nodes to the training set [4, 5]
    ### pseudo_labels has the true label for the train and val nodes and predicted labels for the top k test nodes.
    ### idx_train has the index of the pseudo labels index
    return idx_train, pseudo_labels 
    