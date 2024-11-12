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


def _batch_generator(node_data, class_labels, obj_data, adj_data, selected_idx_list, batch_size):

    train_list =[]
    if obj_data is not None:
        class_data = np.concatenate([class_labels[selected_idx_list], obj_data[selected_idx_list]], axis=-1)
    else:
        class_data = class_labels[selected_idx_list]
    for _, batch in enumerate(zip(copy.deepcopy(node_data[selected_idx_list]),
                                    copy.deepcopy(class_data),
                                    copy.deepcopy(adj_data[selected_idx_list]))):
        if torch.cuda.is_available():
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
            train_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                                    y = torch.from_numpy(batch[1]).cuda().float(),
                                    edge_index = edge_index
                                        )
                                )
        else:
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).float())
            train_list.append(Data(x = torch.from_numpy(batch[0]).float(),
                                    y = torch.from_numpy(batch[1]).float(),
                                    edge_index = edge_index
                                        )
                                )

    return DataLoader(train_list, batch_size=batch_size)


def load_data(obj_data_postfix, part_data_post_fix, file_postfix, seed, batch_size, validation=True):

    
    random.seed(seed)
    torch.manual_seed(seed)

    if validation:
        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train'+part_data_post_fix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_train = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train'+obj_data_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_obj_train = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            class_v = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_train'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            adj_train = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train_val'+part_data_post_fix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_train_val = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train_val'+obj_data_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_obj_train_val = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v_val'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            class_v_val = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_train_val'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            adj_train_val = pickle.load(pickle_file)

        X_train[X_train<=0] = 0
        X_train_val[X_train_val<=0] = 0

        X_train[X_train>=1] = 1
        X_train_val[X_train_val>=1] = 1
    
        X_obj_train[X_obj_train<=0] = 0
        X_obj_train_val[X_obj_train_val<=0] = 0

        X_obj_train[X_obj_train>=1] = 1
        X_obj_train_val[X_obj_train_val>=1] = 1
        train_idx = np.random.randint(1,len(X_train),len(X_train))
        val_idx = np.random.randint(1,len(X_train_val),len(X_train_val))
    

        batch_train_loader = _batch_generator(
            node_data=X_train,
            class_labels=class_v,
            obj_data=X_obj_train,
            adj_data=adj_train,
            selected_idx_list=train_idx,
            batch_size=batch_size)
        batch_val_loader = _batch_generator(
            node_data=X_train_val,
            class_labels=class_v_val,
            obj_data=X_obj_train_val,
            adj_data=adj_train_val,
            selected_idx_list=val_idx,
            batch_size=batch_size)
    
    else:
        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_test'+part_data_post_fix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_test = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_test'+obj_data_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            X_obj_test = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_test'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            adj_test = pickle.load(pickle_file)

        outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v_test'+file_postfix+'.np'
        with open(outfile, 'rb') as pickle_file:
            class_v_test = pickle.load(pickle_file)
        test_idx = np.array(list(range(len(X_test))))
        X_test[X_test<=0] = 0

        batch_val_loader = _batch_generator(
            node_data=X_test,
            class_labels=class_v_test,
            obj_data=X_obj_test,
            adj_data=adj_test,
            selected_idx_list=test_idx,
            batch_size=batch_size)

    return [batch_val_loader]


def load_data_obj_with_parts(obj_data_postfix, part_data_post_fix, file_postfix, seed, batch_size, validation=True):
    
    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_train = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_train = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        class_v = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_train'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train_val'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_train_val = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_train_val'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_train_val = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v_val'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        class_v_val = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_train_val'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train_val = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_test'+part_data_post_fix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_test = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/X_test'+obj_data_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        X_obj_test = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/adj_test'+file_postfix+'.np'
    with open(outfile, 'rb') as pickle_file:
        adj_test = pickle.load(pickle_file)

    outfile = '/Users/amrutamuthal/Documents/training_data/layout_gen/class_v'+file_postfix+'.np'
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
    
    temp = np.pad(X_obj_train, ((0, 0), (1, 0)), 'constant', constant_values=(1))
    temp = np.expand_dims(temp, 1)
    X_train = np.concatenate([temp, X_train], axis=1)
    adj_train = np.pad(adj_train, ((0,0), (1,0), (1,0)), 'constant', constant_values=(0))
    adj_train[:,0 ,0] = 1.
    
    temp = np.pad(X_obj_train_val, ((0, 0), (1, 0)), 'constant', constant_values=(1))
    temp = np.expand_dims(temp, 1)
    X_train_val = np.concatenate([temp, X_train_val], axis=1)
    adj_train_val = np.pad(adj_train, ((0,0), (1,0), (1,0)), 'constant', constant_values=(0))
    adj_train_val[:,0 ,0] = 1.
    
    temp = np.pad(X_obj_test, ((0, 0), (1, 0)), 'constant', constant_values=(1))
    temp = np.expand_dims(temp, 1)
    X_test = np.concatenate([temp, X_test], axis=1)
    adj_test = np.pad(adj_test, ((0,0), (1,0), (1,0)), 'constant', constant_values=(0))
    adj_test[:,0 ,0] = 1.
    
    random.seed(seed)
    train_idx = np.random.randint(1,len(X_train),len(X_train))
    val_idx = np.random.randint(1,len(X_train_val),len(X_train_val))
    test_idx = np.random.randint(1,len(X_test),len(X_test))
    
    torch.manual_seed(seed)

    if validation:

        batch_train_loader = _batch_generator(
            node_data=X_train,
            class_labels=class_v,
            obj_data=X_obj_train,
            adj_data=adj_train,
            selected_idx_list=train_idx,
            batch_size=batch_size)
        batch_val_loader = _batch_generator(
            node_data=X_train_val,
            class_labels=class_v_val,
            obj_data=X_obj_train_val,
            adj_data=adj_train_val,
            selected_idx_list=val_idx,
            batch_size=batch_size)
    
    else:
        batch_train_only_loader = _batch_generator(
            node_data=X_train,
            class_labels=class_v,
            obj_data=X_obj_train,
            adj_data=adj_train,
            selected_idx_list=train_idx,
            batch_size=batch_size)
        batch_val_only_loader = _batch_generator(
            node_data=X_train_val,
            class_labels=class_v_val,
            obj_data=X_obj_train_val,
            adj_data=adj_train_val,
            selected_idx_list=val_idx,
            batch_size=batch_size)

        batch_train_loader = data_utils.ConcatDataset([batch_train_only_loader, batch_val_only_loader])
        batch_val_loader = _batch_generator(
            node_data=X_test,
            class_labels=class_v_test,
            obj_data=X_obj_test,
            adj_data=adj_test,
            selected_idx_list=test_idx,
            batch_size=batch_size)

    return batch_train_loader, batch_val_loader