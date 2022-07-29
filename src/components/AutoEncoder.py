import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from components.Encoder import GCNEncoder, GATEncoder
from components.Decoder import Decoder

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        

class GCNAutoEncoder(nn.Module):
    
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
                 return_label_loss=False,
                 dynamic_margin=False,
                 output_log=False
                ):
        
        super(GCNAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.num_obj_classes = num_obj_classes
        self.encoder = GCNEncoder(latent_dims,
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
                               label_size,
                               output_log
                              )
        
        self.dynamic_margin = dynamic_margin
        self.return_label_loss = return_label_loss
        
        if self.dynamic_margin:
            self.margin_layer = nn.Linear(2*bbx_size, 2)
            self.margin_activation = nn.Sigmoid()
        
    def forward(self, E, X , nodes, obj_class):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
        # The below 2 lines have been commented to compare reconstructions
#         epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
#         z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        z_latent=z_mean
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
#         x_bbx, x_lbl, _, _ = self.decoder(conditioned_z)
        x_bbx,x_lbl, _, _ = self.decoder(conditioned_z)
        margin=None
        if self.dynamic_margin:
#             print('Hello1')

            # change final shape of X_reshaped here for running the alternate experiments
#             X_reshaped = torch.reshape(X, (batch_size, 24, 5))
            X_reshaped = torch.reshape(X, (batch_size,self.num_nodes, 5))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            if self.return_label_loss:
#                 print('Hello2')
                return x_bbx, x_lbl, z_mean, z_logvar, margin
            else:
#                 print('Hello5')
                return x_bbx,z_mean, z_logvar, margin
    
        if self.return_label_loss:
#             print('Hello3')
            return x_bbx, x_lbl, z_mean, z_logvar , margin
        else:
#             print('Hello4')
            return x_bbx,z_mean, z_logvar , margin


class GCNAutoEncoder_Combined_Parts(nn.Module):
    
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
                 return_label_loss=False,
                 dynamic_margin=False,
                 output_log=False
                ):
        
        super(GCNAutoEncoder_Combined_Parts, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.num_obj_classes = num_obj_classes
        self.encoder = GCNEncoder(latent_dims,
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
                               label_size,
                               output_log
                              )
        
        self.dynamic_margin = dynamic_margin
        self.return_label_loss = return_label_loss
        
        if self.dynamic_margin:
            self.margin_layer = nn.Linear(2*bbx_size, 2)
            self.margin_activation = nn.Sigmoid()
        
    def forward(self, E, X , nodes, obj_class):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
        # The below 2 lines have been commented to compare reconstructions
#         epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
#         z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        z_latent=z_mean
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
#         x_bbx, x_lbl, _, _ = self.decoder(conditioned_z)
        x_bbx,x_lbl, _, _ = self.decoder(conditioned_z)
        margin=None
        if self.dynamic_margin:
#             print('Hello1')

            # change final shape of X_reshaped here for running the alternate experiments
            X_reshaped = torch.reshape(X, (batch_size,self.num_nodes, 5))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            if self.return_label_loss:
#                 print('Hello2')
                return x_bbx, x_lbl, z_mean, z_logvar, margin
            else:
#                 print('Hello5')
                return x_bbx,z_mean, z_logvar, margin
    
        if self.return_label_loss:
#             print('Hello3')
            return x_bbx, x_lbl, z_mean, z_logvar , margin
        else:
#             print('Hello4')
            return x_bbx,z_mean, z_logvar , margin






class GATAutoEncoder(nn.Module):
    
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
        
        super(GATAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.num_obj_classes = num_obj_classes
        self.encoder = GATEncoder(latent_dims,
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
        epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
        z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size,self.num_nodes))
        obj_class = torch.reshape(obj_class,(batch_size,self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, x_edge, class_pred = self.decoder(conditioned_z)
        
        return x_bbx, x_lbl, x_edge, class_pred, z_mean, z_logvar
