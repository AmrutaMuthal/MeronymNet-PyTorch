import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

def kl_loss(z_mean,z_logvar):
    
    loss = torch.mean(0.5 * torch.sum((torch.square(z_mean) +
                                      torch.square(torch.exp(z_logvar)) - 
                                      2*(z_logvar) - 1), dim=-1
                                     )
                     )
    
    return loss

def adj_loss(pred_edge, true_edge, batch, num_nodes):
    
    true_edge = to_dense_adj(true_edge, batch=batch, max_num_nodes= num_nodes)
    loss = F.binary_cross_entropy(pred_edge, true_edge, reduction='mean')
    
    return loss

def bbox_loss(pred_box, true_box):
    
    # IOU loss
    smooth_1 = torch.tensor([15]).cuda()
    smooth_2 = torch.tensor([1e-08]).cuda()
    zero = torch.tensor([0.0]).cuda()
    one = torch.tensor([1.0]).cuda()
    true_box = torch.reshape(true_box,pred_box.shape)
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    pred_box = mask*pred_box
    x1g, y1g, x2g, y2g = torch.tensor_split(true_box, 4, dim=-1)
    x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)
    
    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g
    
    interArea = (torch.maximum(zero,
                               (xB - xA + torch.maximum(smooth_1*w,one))) 
                 * torch.maximum(zero, (yB - yA +torch.maximum(smooth_1*h,one))))
    boxAArea = (torch.maximum(zero, 
                              (x2g - x1g + torch.maximum(smooth_1*w,one))) 
                * torch.maximum(zero,
                                (y2g - y1g +torch.maximum(smooth_1*h,one))))
    boxBArea = (torch.maximum(zero, 
                              (x2 - x1 + torch.maximum(smooth_1*w,one))) 
                * torch.maximum(zero,
                                (y2 - y1 + torch.maximum(smooth_1*h,one))))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea
    iou_loss = -torch.log(iouk + smooth_2)
    iou_loss = torch.mean(iou_loss)#/torch.sum(mask)
    
    # Box regression loss
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1)
    reg_loss = torch.sum(reg_loss,dim = -1)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    
    # Pairwise box regression loss
    pair_mse_true = torch.cdist(true_box, true_box)
    pair_mse_pred = torch.cdist(pred_box, pred_box)
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
    
    return iou_loss + reg_loss + pair_loss

def bbox_loss_hw(pred_box, true_box):
    
    # IOU loss
    smooth_1 = torch.tensor([10]).cuda()
    smooth_2 = torch.tensor([1e-08]).cuda()
    zero = torch.tensor([0.0]).cuda()
    one = torch.tensor([0.7]).cuda()
    true_box = torch.reshape(true_box,pred_box.shape)
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    pred_box = mask*pred_box
    xg, yg, wg, hg = torch.tensor_split(true_box, 4, dim=-1)
    xp, yp, wp, hp = torch.tensor_split(pred_box, 4, dim=-1)
    
    xA = torch.maximum(xg, xp)
    yA = torch.maximum(yg, yp)
    xB = torch.minimum(xg+wg, xp+hp)
    yB = torch.minimum(yg+hg, yp+hp)
    
    interArea = (torch.maximum(zero,
                               (xB - xA + torch.maximum(smooth_1*wg,one))) 
                 * torch.maximum(zero, (yB - yA +torch.maximum(smooth_1*hg,one))))
    boxAArea = (torch.maximum(zero, 
                              (wg + torch.maximum(smooth_1*wg,one))) 
                * torch.maximum(zero,
                                (hg +torch.maximum(smooth_1*hg,one))))
    boxBArea = (torch.maximum(zero, 
                              (wp + torch.maximum(smooth_1*wg,one))) 
                * torch.maximum(zero,
                                (hp + torch.maximum(smooth_1*hg,one))))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea
    iou_loss = -torch.log(iouk + smooth_2)
    iou_loss = torch.mean(iou_loss)#/torch.sum(mask)
    
    # Box regression loss
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1)
    reg_loss = torch.sum(reg_loss,dim = -1)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    
    # Pairwise box regression loss
    pair_mse_true = torch.cdist(true_box, true_box)
    pair_mse_pred = torch.cdist(pred_box, pred_box)
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
    
    return iou_loss + reg_loss + pair_loss
    
def node_loss(pred_nodes, true_nodes):
    
    true_nodes = torch.reshape(true_nodes, pred_nodes.shape)
    loss = F.binary_cross_entropy(pred_nodes, true_nodes, reduction='mean')
    
    return loss

def class_loss(pred_class, true_class):
    
    true_class = torch.reshape(true_class, pred_class.shape)
    loss = F.cross_entropy(pred_class,
                           torch.argmax(true_class, dim = -1),
                           reduction='mean')
    
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
    
    