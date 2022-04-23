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
                 label_size=1,
                 output_log=False,
                 predict_parts=False,
                 predict_edges=False,
                 predict_class=False
                 ):
        super(Decoder, self).__init__()
       
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.class_size = class_size
        self.label_size = label_size
        self.output_log = output_log
        self.predict_parts=predict_parts
        self.predict_edges = predict_edges
        self.predict_class = predict_class
        self.dense1 = nn.Linear(latent_dims + num_nodes + class_size,128)  
        self.dense2 = nn.Linear(128,128)
        self.dense_bbx = nn.Linear(128,num_nodes*bbx_size)
        self.dense_lbl = nn.Linear(128,num_nodes*label_size)
        self.dense_edge = nn.Linear(128,num_nodes*num_nodes)
        self.dense_cls = nn.Linear(128,class_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Softmax()
        self.act3 = nn.ReLU()

    def forward(self, embedding):
        x = self.act1(self.dense1(embedding))
        x = self.act1(self.dense2(x))
        x = self.act1(self.dense2(x))
        
        batch_size = x.shape[0]
        if self.output_log:
            x_bbx = self.act3(self.dense_bbx(x))
        else:
            x_bbx = self.act1(self.dense_bbx(x))
        x_bbx = torch.reshape(x_bbx,[batch_size, self.num_nodes, self.bbx_size])
        
        x_lbl=None
        if self.predict_parts:
            x_lbl = self.act1(self.dense_lbl(x))
            x_lbl = torch.reshape(x_lbl,[batch_size, self.num_nodes, self.label_size])
        
        x_edge = None
        if self.predict_edges:
            x_edge = self.act1(self.dense_edge(x))
            x_edge = torch.reshape(x_edge,[batch_size, self.num_nodes, self.num_nodes])
        
        class_pred = None
        if self.predict_class:
            class_pred = self.act2(self.dense_cls(x))
              
        return x_bbx, x_lbl, x_edge, class_pred
