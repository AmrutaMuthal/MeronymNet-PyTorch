import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from components.AutoEncoder import GCNAutoEncoder
from components.DenseAutoencoder import Encoder, Decoder
from components.Encoder import GCNEncoder, GATEncoder
from components.Decoder import Decoder as GCNDecoder
    
    
class TwoStageAutoEncoder(nn.Module):
    
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
                 hidden3=128,
                 dense_hidden1=16,
                 dense_hidden2=32,
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False
                ):
        
        super(TwoStageAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.num_obj_classes = num_obj_classes
        self.gcn_encoder = GCNEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  label_size,
                                  num_obj_classes,
                                  hidden1,
                                  hidden2,
                                  hidden3,
                                 )
        
        self.dynamic_margin = dynamic_margin
        if self.dynamic_margin:
            self.margin_layer = nn.Linear(2*bbx_size, 2)
            self.margin_activation = nn.Sigmoid()
            
        if area_encoding:
            bbx_size-=1
        self.gcn_decoder = GCNDecoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size,
                               output_log=False,
                               predict_edges=False,
                               predict_class=False,
                               object_bbox=True
                              )
        
        self.dense_encoder = Encoder(latent_dims,
                               bbx_size,
                               hidden1,
                               hidden2,
                              )
        self.dense_decoder = Decoder(latent_dims,
                               bbx_size,
                               hidden1,
                               hidden2,
                              )
        
        
        
    def forward(self, E, X_part, X_obj , nodes, obj_class, variational=False):
        
        latent_obj = self.dense_encoder(X_obj)
        
        x_obj_bbx = self.dense_decoder(latent_obj)
        z_mean, z_logvar = self.gcn_encoder(E, X_part, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
        if variational:
            epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
            z_latent = z_mean + epsilon*torch.exp(z_logvar)
        else:
            z_latent = z_mean            
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        conditioned_z = torch.cat([latent_obj, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, _, _ = self.gcn_decoder(conditioned_z)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X_part, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            return x_bbx, x_obj_bbx, x_lbl, z_mean, z_logvar, margin

        return x_bbx, x_obj_bbx, x_lbl, z_mean, z_logvar


