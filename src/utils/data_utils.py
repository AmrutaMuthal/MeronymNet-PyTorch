import numpy as np
import pickle
import random
import copy

import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader
import torch.utils.data as data_utils

def load_data(obj_data_postfix, part_data_post_fix, file_postfix, seed, batch_size, validation=True):
    
    outfile = 'D:/meronym_data/X_train'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_train = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_train'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_train = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        class_v = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_train'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_train_val'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_train_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_train_val'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_train_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v_val'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        class_v_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_train_val'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_test'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_test = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_test'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_test = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_test'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_test = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        class_v_test = pickle.load(pickle_file)
        
    X_train[X_train<=0] = 0
    X_train_val[X_train_val<=0] = 0
    X_test[X_test<=0] = 0

    X_train[X_train>=1] = 1
    X_train_val[X_train_val>=1] = 1
    X_test[X_test>=1] = 1

    X_obj_train[X_obj_train<=0] = 0
    X_obj_train_val[X_obj_train_val<=0] = 0
    X_obj_test[X_obj_test<=0] = 0

    X_obj_train[X_obj_train>=1] = 1
    X_obj_train_val[X_obj_train_val>=1] = 1
    X_obj_test[X_obj_test>=1] = 1
    
    random.seed(seed)
    train_idx = np.random.randint(1,len(X_train),len(X_train))
    val_idx = np.random.randint(1,len(X_train_val),len(X_train_val))
    test_idx = np.random.randint(1,len(X_test),len(X_test))
    
    torch.manual_seed(seed)

    if validation:
        train_list =[]
        for idx, batch in enumerate(zip(copy.deepcopy(X_train[train_idx]),
                                        copy.deepcopy(np.concatenate([class_v[train_idx], X_obj_train[train_idx]], axis=-1)),
                                        copy.deepcopy(adj_train[train_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
            train_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                                   y = torch.from_numpy(batch[1]).cuda().float(),
                                   edge_index = edge_index
                                        )
                             )

        batch_train_loader = DataLoader(train_list, batch_size=batch_size)

        val_list = []
        for idx, batch in enumerate(zip(copy.deepcopy(X_train_val[val_idx]),
                                        copy.deepcopy(np.concatenate([class_v_val[val_idx], X_obj_train_val[val_idx]], axis=-1)),
                                        copy.deepcopy(adj_train_val[val_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
            val_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                                 y = torch.from_numpy(batch[1]).cuda().float(),
                                 edge_index = edge_index
                                        )
                             )
        batch_val_loader = DataLoader(val_list, batch_size=batch_size)
    
    else:
        train_list =[]
        for idx, batch in enumerate(zip(copy.deepcopy(X_train[train_idx]),
                                        copy.deepcopy(np.concatenate([class_v[train_idx], X_obj_train[train_idx]], axis=-1)),
                                        copy.deepcopy(adj_train[train_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda())
            train_list.append(Data(x = torch.from_numpy(batch[0]).cuda(),
                                   y = torch.from_numpy(batch[1]).cuda(),
                                   edge_index = edge_index
                                        )
                             )

        for idx, batch in enumerate(zip(copy.deepcopy(X_train_val[val_idx]),
                                        copy.deepcopy(np.concatenate([class_v_val[val_idx], X_obj_train_val[val_idx]], axis=-1)),
                                        copy.deepcopy(adj_train_val[val_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda())
            train_list.append(Data(x = torch.from_numpy(batch[0]).cuda(),
                                 y = torch.from_numpy(batch[1]).cuda(),
                                 edge_index = edge_index
                                        )
                             )
        batch_train_loader = DataLoader(train_list, batch_size=batch_size)

        val_list = []
        for idx, batch in enumerate(zip(copy.deepcopy(X_test[test_idx]),
                                        copy.deepcopy(np.concatenate([class_v_test[test_idx], X_obj_test[test_idx]], axis=-1)), 
                                        copy.deepcopy(adj_test[test_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda())
            val_list.append(Data(x = torch.from_numpy(batch[0]).cuda(),
                                 y = torch.from_numpy(batch[1]).cuda(),
                                 edge_index = edge_index
                                        )
                             )
        batch_val_loader = DataLoader(val_list, batch_size=batch_size)

    return batch_train_loader, batch_val_loader