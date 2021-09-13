import numpy as np
import torch
import torch.nn as nn

class Decoder(nn.Module):
    """ Decoder module for BoxGCN-Vae
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 class_size,
                 label_size=1
                 ):
        super(Decoder, self).__init__()
       
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.class_size = class_size
        self.label_size = label_size
        self.dense1 = nn.Linear(latent_dims + num_nodes + class_size,128)  
        self.dense2 = nn.Linear(128,128)
        self.dense_bbx = nn.Linear(128,num_nodes*bbx_size)
        self.dense_lbl = nn.Linear(128,num_nodes*label_size)
        self.dense_edge = nn.Linear(128,num_nodes*num_nodes)
        self.dense_cls = nn.Linear(128,class_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Softmax()

    def forward(self, embedding):
        x = self.act1(self.dense1(embedding))
        x = self.act1(self.dense2(x))
        x = self.act1(self.dense2(x))
        
        x_bbx = self.act1(self.dense_bbx(x))
        x_bbx = torch.reshape(x_bbx,[self.num_nodes,self.bbx_size])
        
        x_lbl = self.act1(self.dense_lbl(x))
        x_lbl = torch.reshape(x_lbl,[self.num_nodes,self.label_size])
        
        x_edge = self.act1(self.dense_edge(x))
        x_edge = torch.reshape(x_edge,[self.num_nodes,self.num_nodes])
        
        class_pred = self.act2(self.dense_cls(x))
              
        return x_bbx, x_lbl, x_edge, class_pred
