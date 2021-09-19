import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from Encoder import Encoder
from Decoder import Decoder

class AutoEncoder(nn.Module):
    
    """ AutoEncoder module for Box-Vae
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 num_obj_classes,
                 label_size=1,
                 hidden1=32,
                 hidden2=16,
                 hidden3=128
                ):
        
        super(AutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.num_obj_classes = num_obj_classes
        self.encoder = Encoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               label_size,
                               num_obj_classes,
                               hidden1,
                               hidden2,
                               hidden3,
                              )
        
        self.decoder = Decoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size)
        
    def forward(self,E, X , nodes, obj_class):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
        epsilon = torch.normal(torch.zeros(z_logvar.shape)).cuda()
        z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size,self.num_nodes))
        obj_class = torch.reshape(obj_class,(batch_size,self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, x_edge, class_pred = self.decoder(conditioned_z)
        
        return x_bbx, x_lbl, x_edge, class_pred, z_mean, z_logvar

    