import numpy as np
import torch
import torch.nn.functional as F

def kl_loss(z_mean,z_logvar):
    
    loss = torch.mean(0.5 * torch.sum((torch.square(z_mean) +
                                      torch.square(torch.exp(z_logvar)) - 
                                      2*(z_logvar) - 1)
                                     )
                     )
    
    return loss

def adj_loss(pred_edge, true_edge):
    
    loss = F.binary_cross_entropy(pred_edge, true_edge, reduction='mean')
    
    return loss

def bbox_loss(pred_box, true_box):
    
    # IOU loss
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    pred_box = mask*pred_box
    x1g, y1g, x2g, y2g = torch.tensor_split(true_box, 4, dim=-1)
    x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)
    
    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    interArea = torch.maximum(torch.tensor([0.0]), (xB - xA + 1)) * torch.maximum(torch.tensor([0.0]), yB - yA + 1)
    boxAArea = (x2g - x1g +1) * (y2g - y1g +1)
    boxBArea = (x2 - x1 +1) * (y2 - y1 +1)
    iouk = (interArea+ 1e-6) / (boxAArea + boxBArea - interArea+ 1e-6)
    iou_loss = -torch.log(iouk)
    iou_loss = torch.sum(iou_loss)/torch.sum(mask)
    
    # Box regression loss
    reg_loss = F.mse_loss(pred_box, true_box, reduction='sum')/torch.sum(mask)
    
    # Pairwise box regression loss
    pair_mse_true = []
    pair_mse_pred = []
    true_unstacked = torch.unbind(true_box)
    pred_unstacked = torch.unbind(pred_box)
    
    for i in range(len(true_unstacked)):
        
        for j in range(i, len(true_unstacked)):
            pair_mse_true.append(F.mse_loss(true_unstacked[i],true_unstacked[j]))
            pair_mse_pred.append(F.mse_loss(pred_unstacked[i],pred_unstacked[j]))
        
        pair_loss = F.mse_loss(torch.stack(pair_mse_pred),
                               torch.stack(pair_mse_true))/torch.sum(mask)
    
    return iou_loss+reg_loss+pair_loss
    
def node_loss(pred_nodes, true_nodes):
    
    loss = F.binary_cross_entropy(pred_nodes, true_nodes, reduction='mean')
    
    return loss

def class_loss(pred_class, true_class):
    
    loss = F.binary_cross_entropy(pred_class, true_class, reduction='mean')
    
    return loss

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L
    
    