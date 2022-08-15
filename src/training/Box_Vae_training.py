import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import copy
import torch_geometric.nn as gnn

from torch_geometric.data import DataLoader

import torchvision
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

import losses
from AutoEncoder import AutoEncoder

import sys
import cv2
import os
import math
import pickle

import matplotlib.pyplot as plt

colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702)]
colors = (np.asarray(colors)*255)
canvas_size = 550

def plot_bbx(bbx):
    bbx = bbx*canvas_size
    canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
    for i, coord in enumerate(bbx):
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[i], 6)
    return canvas

def load_data(batch_size: int):
    """ Load train, validation and test sets into data loaders.
    
    Args:
        batch_size: batch size to be used when creating data loader
    
    Returns: train and validation data loaders
    """
    outfile = 'D:/meronym_data/X_train.np'
    with open(outfile, 'rb') as pickle_file:
        X_train = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v.np'
    with open(outfile, 'rb') as pickle_file:
        class_v = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_train.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_train_val.np'
    with open(outfile, 'rb') as pickle_file:
        X_train_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v_val.np'
    with open(outfile, 'rb') as pickle_file:
        class_v_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_train_val.np'
    with open(outfile, 'rb') as pickle_file:
        adj_train_val = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/X_test.np'
    with open(outfile, 'rb') as pickle_file:
        X_test = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/adj_test.np'
    with open(outfile, 'rb') as pickle_file:
        adj_test = pickle.load(pickle_file)

    outfile = 'D:/meronym_data/class_v_test.np'
    with open(outfile, 'rb') as pickle_file:
        class_v_test = pickle.load(pickle_file)

    X_train[X_train<=0] = 0
    X_train_val[X_train_val<=0] = 0
    X_test[X_test<=0] = 0

    X_train[X_train>=1] = 1
    X_train_val[X_train_val>=1] = 1
    X_test[X_test>=1] = 1
    
    random.seed(100)
    train_idx = np.random.randint(1,len(X_train),len(X_train))
    val_idx = np.random.randint(1,len(X_train_val),len(X_train_val))
    test_idx = np.random.randint(1,len(X_test),len(X_test))
    
    seed = 345

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)

    validation = True
    if validation:
        train_list =[]
        for idx, batch in enumerate(zip(copy.deepcopy(X_train[train_idx]),
                                        copy.deepcopy(class_v[train_idx]),
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
                                        copy.deepcopy(class_v_val[val_idx]), 
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
                                        copy.deepcopy(class_v[train_idx]),
                                        copy.deepcopy(adj_train[train_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda())
            train_list.append(Data(x = torch.from_numpy(batch[0]).cuda(),
                                   y = torch.from_numpy(batch[1]).cuda(),
                                   edge_index = edge_index
                                        )
                             )

        for idx, batch in enumerate(zip(copy.deepcopy(X_train_val[val_idx]),
                                        copy.deepcopy(class_v_val[val_idx]), 
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
                                        copy.deepcopy(class_v_test[test_idx]), 
                                        copy.deepcopy(adj_test[test_idx]))):
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda())
            val_list.append(Data(x = torch.from_numpy(batch[0]).cuda(),
                                 y = torch.from_numpy(batch[1]).cuda(),
                                 edge_index = edge_index
                                        )
                             )
        batch_val_loader = DataLoader(val_list, batch_size=batch_size) 
        
        return batch_train_loader, batch_val_loader
    
def main():
    
    latent_dims = 32
    batch_size = 32
    num_nodes = 24
    bbx_size = 4
    num_classes = 10
    label_shape = 1
    nb_epochs = 50
    klw = losses.frange_cycle_linear(nb_epochs*10)
    learning_rate = 0.00002
    
    batch_train_loader, batch_val_loader = load_data(batch_size)


    reconstruction_loss_arr = []
    kl_loss_arr = []
    bbox_loss_arr = []
    adj_loss_arr = []
    node_loss_arr = []

    reconstruction_loss_val_arr = []
    kl_loss_val_arr = []
    bbox_loss_val_arr = []
    adj_loss_val_arr = []
    node_loss_val_arr = []

    vae = AutoEncoder(latent_dims,num_nodes,bbx_size,num_classes,label_shape)
    vae.cuda()
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    writer = SummaryWriter('D:/meronym_data/runs/20k-ld32-lr-'+str(learning_rate)+'-batch-'+str(batch_size))
    icoef = 0

    for epoch in range(nb_epochs):

        batch_loss = 0.0
        batch_kl_loss = 0.0
        batch_bbox_loss = 0.0
        batch_adj_loss = 0.0
        batch_node_loss = 0.0
        images = []

        vae.train()
        i=0
        for train_data in batch_train_loader:

            if torch.cuda.is_available():
                train_data = train_data.cuda()

            node_data_true = train_data.x
            label_true = node_data_true[:,:1]
            class_true = train_data.y
            adj_true = train_data.edge_index
            batch = train_data.batch

            optimizer.zero_grad()

            output = vae(adj_true, node_data_true, label_true , class_true)
            node_data_pred, label_pred, adj_pred, class_pred, z_mean, z_logvar = output

            kl_loss = losses.kl_loss(z_mean, z_logvar)
            adj_loss = losses.adj_loss(adj_pred, adj_true, batch, num_nodes)
            bbox_loss = losses.bbox_loss(node_data_pred, node_data_true[:,1:])
            node_loss = losses.node_loss(label_pred,label_true)
            class_loss = losses.class_loss(class_pred, class_true)

            kl_weight = klw[icoef]

            recostruction_loss = kl_loss*kl_weight + (bbox_loss + node_loss + adj_loss + class_loss)*24*5
            recostruction_loss.backward()
            optimizer.step()

            # print statistics
            batch_loss += recostruction_loss.item()
            batch_kl_loss += kl_loss.item()
            batch_bbox_loss += bbox_loss.item()
            batch_adj_loss += adj_loss.item()
            batch_node_loss += node_loss.item()



            global_step = epoch*len(batch_train_loader)+i
            image_shape = [num_nodes, bbx_size]

            writer.add_scalar("Loss/train/reconstruction_loss", batch_loss/(i+1), global_step)
            writer.add_scalar("Loss/train/kl_loss", batch_kl_loss/(i+1), global_step)
            writer.add_scalar("Loss/train/bbox_loss", batch_bbox_loss/(i+1), global_step)
            writer.add_scalar("Loss/train/adjacency_loss", batch_adj_loss/(i+1), global_step)
            writer.add_scalar("Loss/train/node_loss", batch_node_loss/(i+1), global_step)
            i+=1   
        image = plot_bbx(np.reshape((node_data_true[:24,1:]*label_true[:24]).detach().to("cpu").numpy(),
                                    image_shape)).astype(float)/255
        writer.add_image('train/images/input', image, global_step, dataformats='HWC')
        image = plot_bbx((node_data_pred[0]*label_true[:24]).detach().to("cpu").numpy()).astype(float)/255
        writer.add_image('train/images/generated', image, global_step, dataformats='HWC')

        reconstruction_loss_arr.append(batch_loss/(i+1))
        kl_loss_arr.append(batch_kl_loss/(i+1))
        bbox_loss_arr.append(batch_bbox_loss/(i+1))
        adj_loss_arr.append(batch_adj_loss/(i+1))
        node_loss_arr.append(batch_node_loss/(i+1))


        print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, batch_loss/(i+1) ))

        batch_loss = 0.0
        batch_kl_loss = 0.0
        batch_bbox_loss = 0.0
        batch_adj_loss = 0.0
        batch_node_loss = 0.0
        images = []
        vae.eval()
        for i, val_data in enumerate(batch_val_loader, 0):
            val_data.cuda()
            node_data_true = val_data.x
            label_true = node_data_true[:,:1]
            class_true = val_data.y
            adj_true = val_data.edge_index
            batch = val_data.batch

            kl_weight = klw[icoef]

            output = vae(adj_true, node_data_true, label_true , class_true)
            node_data_pred, label_pred, adj_pred, class_pred, z_mean, z_logvar = output

            kl_loss = losses.kl_loss(z_mean, z_logvar)
            adj_loss = losses.adj_loss(adj_pred, adj_true, batch, num_nodes)
            bbox_loss = losses.bbox_loss(node_data_pred, node_data_true[:,1:])
            node_loss = losses.node_loss(label_pred,label_true)
            class_loss = losses.class_loss(class_pred, class_true)

            recostruction_loss = kl_loss*kl_weight + (bbox_loss + node_loss + adj_loss + class_loss)*24*5

            batch_loss += recostruction_loss.item()
            batch_kl_loss += kl_loss.item()
            batch_bbox_loss += bbox_loss.item()
            batch_adj_loss += adj_loss.item()
            batch_node_loss += node_loss.item()

        image = plot_bbx(np.reshape((node_data_true[:24,1:]*label_true[:24]).detach().to("cpu").numpy(),
                                    image_shape)).astype(float)/255
        writer.add_image('val/images/input', image, (epoch+1)*len(batch_train_loader), dataformats='HWC')
        image = plot_bbx((node_data_pred[0]*label_true[:24]).detach().to("cpu").numpy()).astype(float)/255
        writer.add_image('val/images/generated', image, (epoch+1)*len(batch_train_loader), dataformats='HWC')

        reconstruction_loss_val_arr.append(batch_loss/(i+1))
        kl_loss_val_arr.append(batch_kl_loss/(i+1))
        bbox_loss_val_arr.append(batch_bbox_loss/(i+1))
        adj_loss_val_arr.append(batch_adj_loss/(i+1))
        node_loss_val_arr.append(batch_node_loss/(i+1))

        writer.add_scalar("Loss/val/reconstruction_loss", batch_loss/(i+1), global_step)
        writer.add_scalar("Loss/val/kl_loss", batch_kl_loss/(i+1), global_step)
        writer.add_scalar("Loss/val/kl_loss", batch_kl_loss/(i+1), global_step)
        writer.add_scalar("Loss/val/bbox_loss", batch_bbox_loss/(i+1), global_step)
        writer.add_scalar("Loss/val/adjacency_loss", batch_adj_loss/(i+1), global_step)
        writer.add_scalar("Loss/val/node_loss", batch_node_loss/(i+1), global_step)

        if kl_loss_arr[-1]>0.5 and abs(bbox_loss_arr[-1] - bbox_loss_val_arr[-1]) < 0.2:
            icoef = icoef + 1  

    writer.flush()
    writer.close()
    print('Finished Training')


if __name__ == "__main__":
    main()
