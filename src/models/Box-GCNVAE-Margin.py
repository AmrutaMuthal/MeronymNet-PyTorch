#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


#get_ipython().run_line_magic('set_env', 'CUDA_LAUNCH_BLOCKING=1')


# In[3]:


import sys
import collections
import cv2
import os
import math
import random
import pickle
import copy
import numpy as np
import pickle
sys.path.append("/home/tanvikamble/MeronymNet-PyTorch/src")


# In[4]:


import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter


# In[5]:


from losses import BoxVAE_losses as loss
from components.AutoEncoder import GCNAutoEncoder
from components.Decoder import Decoder


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


# from mask_generation import masked_sketch


# In[8]:


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
canvas_size = 660

def plot_bbx(bbx):
    bbx = bbx*canvas_size
    canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
    for i, coord in enumerate(bbx):
        x_minp, y_minp,x_maxp , y_maxp= coord
        if [x_minp, y_minp,x_maxp , y_maxp]!=[0,0,0,0]:
            cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[i], 6)
    return canvas

# def plot_bbx(bbx):
#     bbx = bbx*canvas_size
#     canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
#     for i, coord in enumerate(bbx):
#         x, y, w ,h = coord
#         if [x, y, w ,h]!=[0,0,0,0]:
#             cv2.rectangle(canvas, (int(x), int(y)), (int(x + w) , int(y + h) ), colors[i], 6)
#     return canvas


# In[9]:


def inference(decoder, nodes, obj_class, latent_dims, batch_size):

    decoder.cuda()
    z_latent = torch.normal(torch.zeros([batch_size,latent_dims])).cuda()
    nodes = torch.reshape(nodes,(batch_size,decoder.num_nodes))
    obj_class = torch.reshape(obj_class,(batch_size, decoder.class_size))
    conditioned_z = torch.cat([nodes, z_latent],dim=-1)
    conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
    
    x_bbx, x_lbl = decoder(conditioned_z)
    if x_lbl == None:
        return x_bbx,z_latent
    else:
        return x_bbx, x_lbl, z_latent


# In[10]:


outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/X_train.np'
with open(outfile, 'rb') as pickle_file:
    X_train = pickle.load(pickle_file)

outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/class_v.np'
with open(outfile, 'rb') as pickle_file:
    class_v = pickle.load(pickle_file)

outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/adj_train.np'
with open(outfile, 'rb') as pickle_file:
    adj_train = pickle.load(pickle_file)

outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/X_train_val.np'
with open(outfile, 'rb') as pickle_file:
    X_train_val = pickle.load(pickle_file)

outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/class_v_val.np'
with open(outfile, 'rb') as pickle_file:
    class_v_val = pickle.load(pickle_file)
    
outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/adj_train_val.np'
with open(outfile, 'rb') as pickle_file:
    adj_train_val = pickle.load(pickle_file)
    
outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/X_test.np'
with open(outfile, 'rb') as pickle_file:
    X_test = pickle.load(pickle_file)

outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/adj_test.np'
with open(outfile, 'rb') as pickle_file:
    adj_test = pickle.load(pickle_file)
    
outfile = '/home/tanvikamble/MeronymNet-PyTorch/src/processed_data/class_v_test.np'
with open(outfile, 'rb') as pickle_file:
    class_v_test = pickle.load(pickle_file)

X_train[:, :, 4] = X_train[:, :, 4] - X_train[:, :, 2]
X_train[:, :, 3] = X_train[:, :, 3] - X_train[:, :, 1]

X_train_val[:, :, 4] = X_train_val[:, :, 4] - X_train_val[:, :, 2]
X_train_val[:, :, 3] = X_train_val[:, :, 3] - X_train_val[:, :, 1]

X_test[:, :, 4] = X_test[:, :, 4] - X_test[:, :, 2]
X_test[:, :, 3] = X_test[:, :, 3] - X_test[:, :, 1]
# In[11]:


X_train[X_train<=0] = 0
X_train_val[X_train_val<=0] = 0
X_test[X_test<=0] = 0

X_train[X_train>=1] = 1
X_train_val[X_train_val>=1] = 1
X_test[X_test>=1] = 1


# In[ ]:





# In[12]:


random.seed(100)
train_idx = np.random.randint(1,len(X_train),len(X_train))
val_idx = np.random.randint(1,len(X_train_val),len(X_train_val))
test_idx = np.random.randint(1,len(X_test),len(X_test))


# In[13]:


batch_size = 128
seed = 345

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
        edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
        train_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                               y = torch.from_numpy(batch[1]).cuda().float(),
                               edge_index = edge_index
                                    )
                         )
    
    for idx, batch in enumerate(zip(copy.deepcopy(X_train_val[val_idx]),
                                    copy.deepcopy(class_v_val[val_idx]), 
                                    copy.deepcopy(adj_train_val[val_idx]))):
        edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
        train_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                             y = torch.from_numpy(batch[1]).cuda().float(),
                             edge_index = edge_index
                                    )
                         )
    batch_train_loader = DataLoader(train_list, batch_size=batch_size)
    
    val_list = []
    for idx, batch in enumerate(zip(copy.deepcopy(X_test[test_idx]),
                                    copy.deepcopy(class_v_test[test_idx]), 
                                    copy.deepcopy(adj_test[test_idx]))):
        edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
        val_list.append(Data(x = torch.from_numpy(batch[0]).cuda().float(),
                             y = torch.from_numpy(batch[1]).cuda().float(),
                             edge_index = edge_index
                                    )
                         )
    batch_val_loader = DataLoader(val_list, batch_size=batch_size)
    


# In[14]:


del train_list
del val_list
del X_train
del class_v
del adj_train
del X_train_val
del class_v_val
del adj_train_val


# In[15]:


idx = 0
for data in batch_train_loader:
    idx+=1
    print(data.x.shape)
    if idx==3:
        break


# In[28]:


latent_dims = 64
# batch_size = 128
batch_size = 8
num_nodes = 24
bbx_size = 4
num_classes = 10
label_shape = 1
nb_epochs = 250
# nb_epochs = 2
klw = loss.frange_cycle_linear(nb_epochs)
learning_rate = 0.000065
hidden1 = 32
hidden2 = 16
hidden3 = 128
adaptive_margin = True
# adaptive_margin = False
run_prefix = "Using_Metrics"


# In[29]:


import gc
gc.collect()


# In[30]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[31]:


def metrics(class_true,iou_loss,mse_loss,iou_loss_dict,mse_loss_dict):
    class_dict = ['cow', 'sheep', 'bird', 'person', 'cat', 'dog', 'horse', 'aeroplane',
              'motorbike', 'bicycle', 'car']
    batch_size=iou_loss.shape[0]
    for i in range(batch_size):
        class_i = class_dict[int(np.argmax(class_true[10*i:10*(i+1)].detach().to('cpu').numpy()).tolist())]
        if class_i in iou_loss_dict and class_i in mse_loss_dict:
            iou_loss_dict[class_i].append(iou_loss[i])
            mse_loss_dict[class_i].append(mse_loss[i])
        else:
            iou_loss_dict[class_i]=[iou_loss[i]]
            mse_loss_dict[class_i]=[mse_loss[i]]
    return iou_loss_dict,mse_loss_dict


# In[32]:


def append(iou_loss_dict_final,mse_loss_dict_final,iou_loss_dict,mse_loss_dict):
    for class_i in iou_loss_dict:
        if class_i in iou_loss_dict_final:
            iou_loss_dict_final[class_i].append(sum(iou_loss_dict[class_i])/len(iou_loss_dict[class_i]))
            mse_loss_dict_final[class_i].append(sum(mse_loss_dict[class_i])/len(mse_loss_dict[class_i]))
        else:
            iou_loss_dict_final[class_i] = [sum(iou_loss_dict[class_i])/len(iou_loss_dict[class_i])]
            mse_loss_dict_final[class_i] = [sum(mse_loss_dict[class_i])/len(mse_loss_dict[class_i])]
    return iou_loss_dict_final,mse_loss_dict_final


# In[33]:


reconstruction_loss_arr = []
kl_loss_arr = []
bbox_loss_arr = []
adj_loss_arr = []
iou_loss_dict_final={}
mse_loss_dict_final={}
# node_loss_arr = []

reconstruction_loss_val_arr = []
kl_loss_val_arr = []
bbox_loss_val_arr = []
adj_loss_val_arr = []
iou_loss_val_dict_final={}
mse_loss_val_dict_final={}

# node_loss_val_arr = []


vae = GCNAutoEncoder(latent_dims,
                     num_nodes,
                     bbx_size,
                     num_classes,
                     label_shape,
                     hidden1,
                     hidden2,
                     hidden3,
                     dynamic_margin=adaptive_margin)
vae.to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
model_path = ('/home/tanvikamble/MeronymNet-PyTorch/src/model/'+run_prefix+'/GCN-lr-'
                        +str(learning_rate)
                        +'-batch-'+str(batch_size)
                        +'-h1-'+str(hidden1)
                        +'-h2-'+str(hidden2)
                        +'-h3-'+str(hidden3)+'-test')
summary_path = ('/home/tanvikamble/MeronymNet-PyTorch/src/runs/'+run_prefix+'/GCN-lr-'
                        +str(learning_rate)
                        +'-batch-'+str(batch_size)
                        +'-h1-'+str(hidden1)
                        +'-h2-'+str(hidden2)
                        +'-h3-'+str(hidden3)+'-test')
if not os.path.exists(model_path):
    os.makedirs(model_path)
writer = SummaryWriter(summary_path)
icoef = 0

for epoch in range(nb_epochs):

    batch_loss = torch.tensor([0.0])
    batch_kl_loss = torch.tensor([0.0])
    batch_bbox_loss = torch.tensor([0.0])
#     batch_node_loss = torch.tensor([0.0])
    images = []
    iou_loss_dict={}
    mse_loss_dict={}
    iou_loss_val_dict={}
    mse_loss_val_dict={}
    
    vae.train()
    i=0
    for train_data in batch_train_loader:
        node_data_true = train_data.x
        label_true = node_data_true[:,:1]
        class_true = train_data.y
        adj_true = train_data.edge_index
        batch = train_data.batch
        
        for param in vae.parameters():
            param.grad=None
        
        output = vae(adj_true, node_data_true, label_true , class_true)
#         node_data_pred, label_pred, z_mean, z_logvar, margin = output
#         print(len(output))
        node_data_pred, z_mean, z_logvar, margin = output
        
        kl_loss = loss.kl_loss(z_mean, z_logvar)
        
        bbox_loss = loss.bbox_loss(node_data_pred, node_data_true[:,1:], margin)
        
        iou_loss,mse_loss=loss.calc_losses(node_data_pred, node_data_true[:,1:])
        
        iou_loss_dict,mse_loss_dict=metrics(class_true,iou_loss,mse_loss,iou_loss_dict,mse_loss_dict)
        
#         node_loss = loss.node_loss(label_pred,label_true)
        
#         kl_weight = klw[icoef]
        kl_weight=0
        if kl_weight>0:
            reconstruction_loss = kl_loss*kl_weight + (bbox_loss)*24*5
        else:
            reconstruction_loss = (bbox_loss)*24*5
            
        reconstruction_loss.backward()
        
        optimizer.step()
        
        i+=1
      
        batch_loss += reconstruction_loss
        batch_kl_loss += kl_loss
        batch_bbox_loss += bbox_loss
#         batch_node_loss += node_loss
    
        if i%200==0:
            print(i)
            global_step = epoch*len(batch_train_loader)+i
            
            writer.add_scalar("Loss/train/reconstruction_loss", batch_loss.item()/(i+1), global_step)
            writer.add_scalar("Loss/train/kl_loss", batch_kl_loss.item()/(i+1), global_step)
            writer.add_scalar("Loss/train/bbox_loss", batch_bbox_loss.item()/(i+1), global_step)
#             writer.add_scalar("Loss/train/node_loss", batch_node_loss.item()/(i+1), global_step)
    iou_loss_dict_final,mse_loss_dict_final = append(iou_loss_dict_final,mse_loss_dict_final,iou_loss_dict,mse_loss_dict)
            
    global_step = epoch*len(batch_train_loader)+i
    image_shape = [num_nodes, bbx_size]

    image = plot_bbx(np.reshape((node_data_true[:24,1:]*label_true[:24]).detach().to("cpu").numpy(),
                                image_shape)).astype(float)/255
    writer.add_image('train/images/input', image, global_step, dataformats='HWC')
    image = plot_bbx((node_data_pred[0]*label_true[:24]).detach().to("cpu").numpy()).astype(float)/255
    writer.add_image('train/images/generated', image, global_step, dataformats='HWC')
    
    reconstruction_loss_arr.append(batch_loss.detach().item()/(i+1))
    kl_loss_arr.append(batch_kl_loss.detach().item()/(i+1))
    bbox_loss_arr.append(batch_bbox_loss.detach().item()/(i+1))
#     node_loss_arr.append(batch_node_loss.detach().item()/(i+1))
    
    print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, batch_loss/(i+1) ))
    
    batch_loss = torch.tensor([0.0])
    batch_kl_loss = torch.tensor([0.0])
    batch_bbox_loss = torch.tensor([0.0])
#     batch_node_loss = torch.tensor([0.0])
    images = []
    vae.eval()
    for i, val_data in enumerate(batch_val_loader, 0):
        node_data_true = val_data.x
        label_true = node_data_true[:,:1]
        class_true = val_data.y
        adj_true = val_data.edge_index
        batch = val_data.batch
        
#         kl_weight = klw[icoef]
        kl_weight=0
        
        output = vae(adj_true, node_data_true, label_true , class_true)
        
#         node_data_pred, label_pred, z_mean, z_logvar, margin = output
        node_data_pred, z_mean, z_logvar, margin = output
        kl_loss = loss.kl_loss(z_mean, z_logvar)
        bbox_loss = loss.bbox_loss(node_data_pred, node_data_true[:,1:], margin)
        
        iou_loss,mse_loss=loss.calc_losses(node_data_pred, node_data_true[:,1:])
        
        iou_loss_val_dict,mse_loss_val_dict=metrics(class_true,iou_loss,mse_loss,iou_loss_val_dict,mse_loss_val_dict)
#         node_loss = loss.node_loss(label_pred,label_true)
        
        reconstruction_loss = kl_loss*kl_weight + (bbox_loss)*24*5
        
        batch_loss += reconstruction_loss
        batch_kl_loss += kl_loss
        batch_bbox_loss += bbox_loss
#         batch_node_loss += node_loss
    
    iou_loss_val_dict_final,mse_loss_val_dict_final = append(iou_loss_val_dict_final,mse_loss_val_dict_final,iou_loss_val_dict,mse_loss_val_dict)
    image = plot_bbx(np.reshape((node_data_true[:24,1:]*label_true[:24]).detach().to("cpu").numpy(),
                                image_shape)).astype(float)/255
    writer.add_image('val/images/input', image, global_step, dataformats='HWC')
    image = plot_bbx((node_data_pred[0]*label_true[:24]).detach().to("cpu").numpy()).astype(float)/255
    writer.add_image('val/images/generated', image, global_step, dataformats='HWC')

    reconstruction_loss_val_arr.append(batch_loss.detach().item()/(i+1))
    kl_loss_val_arr.append(batch_kl_loss.detach().item()/(i+1))
    bbox_loss_val_arr.append(batch_bbox_loss.detach().item()/(i+1))
#     node_loss_val_arr.append(batch_node_loss.detach().item()/(i+1))
    
    writer.add_scalar("Loss/val/reconstruction_loss", batch_loss.detach()/(i+1), global_step)
    writer.add_scalar("Loss/val/kl_loss", batch_kl_loss.detach()/(i+1), global_step)
    writer.add_scalar("Loss/val/bbox_loss", batch_bbox_loss.detach()/(i+1), global_step)
#     writer.add_scalar("Loss/val/node_loss", batch_node_loss.detach()/(i+1), global_step)
       
    if epoch%50 == 0:
        torch.save(vae.state_dict(), model_path + '/model_weights.pth')

#     if kl_loss_arr[-1]>0.5 and abs(bbox_loss_arr[-1] - bbox_loss_val_arr[-1]) < 0.012 and bbox_loss_arr[-1]<0.08 and epoch>60:
#         icoef = icoef + 1  

torch.save(vae.state_dict(),model_path + '/model_weights.pth')

for i in range(100):    
    image = plot_bbx(np.reshape((node_data_true[24*i:24*(i+1),1:]*label_true[24*i:24*(i+1)]).detach().to("cpu").numpy(),
                                    image_shape)).astype(float)/255
    writer.add_image('result/images/'+str(i)+'-input', image, global_step, dataformats='HWC')
    image = plot_bbx((node_data_pred[i]*label_true[24*i:24*(i+1)]).detach().to("cpu").numpy()).astype(float)/255
    writer.add_image('result/images/'+str(i)+'-generated', image, global_step, dataformats='HWC')
    
writer.flush()
writer.close()
print('Finished Training')


# In[ ]:


# iou_loss_dict_final={}
# mse_loss_dict_final={}
# iou_loss_val_dict_final={}
# mse_loss_val_dict_final={}
file_name= run_prefix + "_With_Margin_Latest_Update" + "iou_loss_train"
file = open(file_name, 'wb')
pickle.dump(iou_loss_dict_final, file)
file.close()

file_name= run_prefix + "_With_Margin_Latest_Update" + "mse_loss_train"
file = open(file_name, 'wb')
pickle.dump(mse_loss_dict_final, file)
file.close()

file_name= run_prefix + "_With_Margin_Latest_Update" + "iou_loss_val"
file = open(file_name, 'wb')
pickle.dump(iou_loss_val_dict_final, file)
file.close()

file_name= run_prefix + "_With_Margin_Latest_Update" + "mse_loss_val"
file = open(file_name, 'wb')
pickle.dump(mse_loss_val_dict_final, file)
file.close()


# In[20]:


#testing loop
model_path = ('/home/tanvikamble/MeronymNet-PyTorch/src/model/'+run_prefix+'/GCN-lr-'
                        +str(learning_rate)
                        +'-batch-'+str(batch_size)
                        +'-h1-'+str(hidden1)
                        +'-h2-'+str(hidden2)
                        +'-h3-'+str(hidden3)+'-test')
summary_path = ('/home/tanvikamble/MeronymNet-PyTorch/src/runs/'+run_prefix+'/GCN-lr-'
                        +str(learning_rate)
                        +'-batch-'+str(batch_size)
                        +'-h1-'+str(hidden1)
                        +'-h2-'+str(hidden2)
                        +'-h3-'+str(hidden3)+'-test')

class_dict = ['cow', 'sheep', 'bird', 'person', 'cat', 'dog', 'horse', 'aeroplane',
              'motorbike', 'bicycle', 'car']
count_dict = {'cow':0, 'sheep':0, 'bird':0, 'person':0, 'cat':0, 'dog':0, 'horse':0,
              'aeroplane':0, 'motorbike':0, 'bicycle':0, 'car':0}
write_tensorboard = True
if write_tensorboard:
    writer = SummaryWriter(summary_path)

vae = GCNAutoEncoder(latent_dims,num_nodes,bbx_size,num_classes,label_shape,hidden1, hidden2, hidden3, dynamic_margin=adaptive_margin)
vae.load_state_dict(torch.load(model_path+ '/model_weights.pth'))

decoder = vae.decoder
image_shape = [num_nodes, bbx_size]
global_step = 250000
pred_boxes = []
classes = []
for i, val_data in enumerate(batch_val_loader, 0):
    
    val_data.cuda()
    node_data_true = val_data.x
    label_true = node_data_true[:,:1]
    class_true = val_data.y
    val_batch_size = int(class_true.shape[0]/10)
    adj_true = val_data.edge_index
#     output = inference(decoder, label_true , class_true, latent_dims, val_batch_size)
    output = vae(adj_true, node_data_true, label_true , class_true)
    node_data_pred_test = output[0]
    pred_boxes.append((node_data_pred_test*label_true.reshape([int(label_true.shape[0]/24),24,1])).detach().to("cpu").numpy())
    classes.append(class_true.detach().to("cpu").numpy())
    
    if write_tensorboard:
        
        for j in range(int(len(node_data_true)/24)):
            
            obj_class = class_dict[int(np.argmax(class_true[10*j:10*(j+1)].detach().to('cpu').numpy()).tolist())]
            if count_dict[obj_class]<10:
                print(obj_class, count_dict[obj_class])
                image = plot_bbx(np.reshape((node_data_true[24*j:24*(j+1),1:]*label_true[24*j:24*(j+1)]).detach().to("cpu").numpy(),
                                            image_shape)).astype(float)/255
                pred_image = plot_bbx((node_data_pred_test[j]*label_true[24*j:24*(j+1)]).detach().to("cpu").numpy()).astype(float)/255

                writer.add_image('test_result/images/'+obj_class+'/'+str(j)+'-input/', image, global_step, dataformats='HWC')  
                writer.add_image('test_result/images/'+obj_class+'/'+str(j)+'-generated/', pred_image, global_step, dataformats='HWC')
                count_dict[obj_class]+=1

writer.flush()
writer.close()
    


# In[ ]:


#int(np.argmax(class_true[10*j:10*(j+1)].detach().to('cpu').numpy()).tolist())


# In[ ]:


#outfile = 'D:/meronym_data/generate_boxes.npy'
#with open(outfile, 'wb') as pickle_file:
  #  pred_boxes = np.concatenate(pred_boxes)
  #  pickle.dump(pred_boxes, pickle_file)
#outfile = 'D:/meronym_data/generate_boxesobj_class.npy'
#with open(outfile, 'wb') as pickle_file:
 #   pickle.dump(classes,pickle_file)


# In[ ]:


#import gc
#gc.collect()

