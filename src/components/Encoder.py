import numpy as np
import torch
import torch.nn as nn
from GCLayer import GCLayer

class Encoder(nn.Module):
    """ Encoder module for AutoEncoder in BoxGCN-VAE. 
    Args:
        num_nodes: number of nodes in the encoder.
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 label_size,
                 num_obj_classes
                 ):
        super(Encoder, self).__init__()
       
        self.gconv1 = GCLayer(bbx_size+label_size,32)
        self.gconv2 = GCLayer(32,16)
        self.dense_boxes = nn.Linear(bbx_size, 16)
        self.dense_labels = nn.Linear(label_size,16)
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(16*num_nodes,128)
        self.dense2 = nn.Linear(16*num_nodes+num_obj_classes,128)
        self.dense3 = nn.Linear(128,128)
        
        self.latent = nn.Linear(128,latent_dims)

    def forward(self, E, X_data,class_labels):
        
        x = self.gconv1(X_data,E)
        x = self.gconv2(x,E)
        x = torch.flatten(x)
        
        boxes = X_data[:,1:]
        boxes = self.act(self.dense_boxes(boxes))
        
        node_labels = X_data[:,:1]
        node_labels = self.act(self.dense_labels(node_labels))
        
        mix = torch.add(boxes,node_labels)
        mix = torch.flatten(mix)
        mix = self.act(self.dense1(mix))
        
        x = torch.cat([class_labels,x])
        x = self.act(self.dense2(x))
        x = torch.add(x,mix)
        x = self.act(self.dense3(x))
        x = self.act(self.dense3(x))
        
        z_mean = self.act(self.latent(x))
        z_logvar = self.act(self.latent(x))
        
        return z_mean,z_logvar