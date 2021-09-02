import numpy as np
import torch
import torch.nn as nn
from components import Encoder, Decoder

class AutoEncoder(nn.Module):
    
    """ AutoEncoder module for Box-Vae
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 num_obj_classes,
                 label_size=1
                ):
        
        super(AutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.encoder = Encoder(latent_dims,
                               num_nodes,
                               bbx_size)
        
        self.decoder = Decoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size)
        
    def forward(self,E, X , nodes, obj_class):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        z_latent = z_mean + torch.randn(self.latent_dims)*torch.exp(z_logvar)
        x_bbx, x_lbl, x_edge, class_pred = self.decoder(z_latent)
        #true_edge=E, true_node=X, latent_dim,  true_class=nodes, class_vec=class_pred)
        # conditioning has to be added
        return x_bbx, x_lbl, x_edge, class_pred

    