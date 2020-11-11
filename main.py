# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:22:44 2020

@author: John
"""

import math
import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from   tensorboardX import SummaryWriter
from   torch.utils.data  import DataLoader
from torchvision import transforms as T
from dataloader  import CarclassifyLoader
from torch.nn.parallel import DataParallel
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.multiclass import unique_labels
from torchsummary import summary
import csv
from ranger import Ranger
# from ipywidgets import IntProgress

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

# device = [0,1,2]
# HyperParemeters
Batch_size = 64
Learning_rate = 1e-3
Epochs = 200

all_label = []
with open('./cs-t0828-2020-hw1/training_labels.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        all_label.append(row[1])
final_label= sorted(list(set(all_label)))
label_dic = {}
for i in range(len(final_label)):
    label_dic.update({i:final_label[i]})
# Calculate accuracy
def compute_accuracy( predict , gt):
    predict_label = F.softmax(predict , dim = 1).data.numpy()
    correct_prediction = np.equal( np.argmax( predict_label,axis=1), gt)
    accuracy = np.mean( correct_prediction)
    return accuracy


# BasicBlock for ResNet18
class BasicBlock( nn.Module):
    def __init__(self , in_channel , out_channel, strides, downsample = None):
        super(BasicBlock , self).__init__()
        self.scalar = 1
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channel ,out_channel , 
                      kernel_size=(3,3) , stride = strides , padding = (1,1) , bias=False
                      ),
            nn.BatchNorm2d(out_channel , 
                           eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True
                           ),
            nn.ReLU(),
            nn.Conv2d(out_channel ,out_channel 
                      , kernel_size=(3,3) , stride = (1,1) , padding = (1,1) , bias=False
                      ),
            nn.BatchNorm2d(out_channel*self.scalar   , 
                           eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True
                           )
            )
        self.downsample = nn.Sequential()
        if downsample :
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channel ,out_channel*self.scalar , 
                              kernel_size=(1,1) , stride =  (2,2) , bias=False),
                    nn.BatchNorm2d(out_channel*self.scalar)
                              )
    def forward(self , x):
        residual = x
        out = self.basicblock(x)
        out += self.downsample(residual)
        out = F.relu(out)
        return out

# BottleneckBlock for ResNet50
class BottleneckBlock( nn.Module):
    def __init__(self , in_channel , out_channel, strides = 1 , downsample = True ):
        super(BottleneckBlock , self).__init__()
        self.scalar = 4
        self.bottleneckblock = nn.Sequential(
            nn.Conv2d(in_channel ,out_channel , kernel_size=(1,1) , bias=False),
            nn.BatchNorm2d(out_channel , eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True),
            nn.ReLU(),
            nn.Conv2d(out_channel ,out_channel , kernel_size=(3,3) , stride = strides , padding = (1,1) , bias=False),
            nn.BatchNorm2d(out_channel  , eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True),
            nn.ReLU(),
            nn.Conv2d(out_channel ,out_channel*self.scalar , kernel_size=(1,1) , bias=False),
            nn.BatchNorm2d(out_channel*self.scalar   , eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True)
            )
        self.downsample = nn.Sequential()
        if downsample :
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channel ,out_channel*self.scalar , 
                              kernel_size=(1,1) , stride = strides, bias=False),
                    nn.BatchNorm2d(out_channel*self.scalar)
                              )
    def forward(self , x):
        residual = x
        
        out = self.bottleneckblock(x)
        out += self.downsample(residual)
        out = F.relu(out)
        return out

class ResNet( nn.Module):
    def __init__(self, block , residual_times , num_classes , scalar):
        '''
        block : using which block (BasicBlock or BottleneckBlock)
        residual times:  the repeat times of rach conv layer
        num_classes   : how many classes need to classify
        scalar : for ResNet18 is 1 for ResNet50 is 4
        '''
        super(ResNet , self).__init__()
        self.in_channel = [3 , 64 ,64 ,128 ,256,512]
        self.out_channel = [64 ,64 ,128 ,256, 512, num_classes]
        self.residual_times = residual_times
        self.layer = nn.Sequential(
            nn.Conv2d(self.in_channel[0] ,self.out_channel[0] , kernel_size=(7,7) , stride = (2,2) , padding = (3,3) , bias=False),
            nn.BatchNorm2d(self.out_channel[0], eps = 1e-05 , momentum=0.1 , affine = True , track_running_stats= True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.basiclayer = nn.ModuleList()
        stride = 1
        for idx , t in enumerate( self.residual_times,1):
            if idx > 1:
                stride = 2
            # Times of residual block
            for times in range(t):
                # Need downsampling 
                if times == 0 and self.in_channel[idx] != self.out_channel[idx]*scalar: 
                    self.basiclayer.append(block(self.in_channel[idx], self.out_channel[idx],stride, True))
                    stride = 1
                else:
                    self.basiclayer.append(block(self.out_channel[idx]*scalar, self.out_channel[idx],stride, None))
            self.in_channel[idx+1]*= scalar
        self.lastlayer = nn.Sequential(
                nn.AvgPool2d( kernel_size = 7, stride = 1,padding=0),
                nn.Linear(in_features = self.in_channel[-1] , out_features = self.out_channel[-1] , bias = True)
                )
    def forward( self, x):
        x = self.layer(x)
        for module in self.basiclayer:
            x = module(x)
        x = self.lastlayer[0](x)
        x = x.view(-1,self.in_channel[-1])
        x = self.lastlayer[1](x)
        return x

class NetworkV2(nn.Module):
    def __init__(self, num_classes, num_makes, num_types):
        super().__init__()
        # self.base = models.resnet50(pretrained = True)
        self.base = models.resnext50_32x4d(pretrained=True)

        if hasattr(self.base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_types, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        type_fc = self.type_fc(out)

        concat = torch.cat([out, brand_fc, type_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, type_fc
    
def main(mode,n ):
    '''
    mode : "train " or "eval"
    n : use which model  , 0: resnet18 , 1:pretrain_resnet18 , 2:resnet50 , 3:pretrain_resnet50
    ''' 
    device = torch.device("cuda")
    data_train     =  CarclassifyLoader('train')
    data_eval     =  CarclassifyLoader('eval')
    data_test     =  CarclassifyLoader('test')
    
    
    #pretrained resnext50
    pretrain_resnext50 = NetworkV2(196, 49, 22)
    
    #pretrained resnext50
    pretrain_resnet50 = models.resnet50(pretrained = True)
    # Replace the last FC layer
    pretrain_resnet50.fc = nn.Linear(pretrain_resnet50.fc.in_features, 196)
    
    
    # optimize trainable parameters
    optimizer_pre_next50 = Ranger(pretrain_resnext50.parameters())

    optimizer_pre_50 = optim.SGD(pretrain_resnet50.parameters(), lr=Learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # Cross Entropy
    loss_func = nn.CrossEntropyLoss()  
    write = [ False , False]
    # Tensorboard path
    if mode == "train":
        write[n] = True
    else :
        write[n] = False
    tb_next_50 = SummaryWriter(log_dir = './tensorboard/ResNext50_Pretrained',write_to_disk=write[n])
    tb_pre_50 = SummaryWriter(log_dir = './tensorboard/ResNet50_Pretrained',write_to_disk=write[n])
    
    # Total model & optimizer & zip together
    model_total     = [pretrain_resnext50,pretrain_resnet50]
    optimizer_total = [optimizer_pre_next50 ,optimizer_pre_50  ]
    tb_total        = [tb_next_50 , tb_pre_50]
    epoch_total = [ 1000 , 1000]
    name_total      = ["ResNext50_Pretrained" , "ResNet50_Pretrained" ]
    # ziplist = zip(model_total  ,optimizer_total , tb_total,epoch_total,name_total )   
    if n == 0:
        model = pretrain_resnext50
        optimizer = optimizer_pre_next50
    elif n == 1:
        model = pretrain_resnet50
        optimizer = optimizer_pre_50

    if mode == "train":
        model = DataParallel(model) 
    # training and testing
    path_checkpoint = "./"+name_total[n]
    if not os.path.isdir(path_checkpoint):
        os.makedirs(path_checkpoint)
    # train mode
    if mode == "train":
        # current epoch
        cur_e = 0


        # Pytorch DataLoader
        loader_train = DataLoader(
            # Torch TensorDataset format    
            dataset = data_train,
            # Mini batch size
            batch_size = Batch_size,
            # training : random shuffle
            shuffle    = True,
            num_workers = 4
            )
        loader_eval = DataLoader(
                # Torch TensorDataset format    
                dataset = data_eval,
                # Mini batch size
                batch_size = 1,
                # training : random shuffle
                num_workers = 4
                )
        cnt=0
        path_tb = "./tensorboard/"
        tb_train = SummaryWriter(log_dir = path_tb+name_total[n]+"/train")
        tb_test = SummaryWriter(log_dir = path_tb+name_total[n]+"/test")
        model.cuda()
        for e in range(cur_e,epoch_total[n]+1):
            model.train()
            acc_train = 0          
            loss_train = 0
            for iter_train, (batch_name, img, final_lab, make_lab, type_lab) in enumerate(loader_train):    
                img   = img.float().to(device)
                final_lab   = final_lab.long()  
    
                final_y, make_y, type_y = model(img)         
                final_y = final_y.cpu()
                make_y  = make_y.cpu()
                type_y = type_y.cpu()
                
                loss_final = loss_func(final_y , final_lab ) 
                loss_make = loss_func(make_y  , make_lab )
                loss_type = loss_func(type_y  , type_lab )
                loss = loss_final + 0.1*loss_make + 0.1*loss_type
                # clear gradients for this training step
                optimizer.zero_grad()     
                # backpropagation, compute gradients
                loss.backward()           
                # apply gradients
                optimizer.step()               
                acc_train_iter = compute_accuracy(final_y, final_lab.data.numpy())
                cnt += 1
                print("Iteration:",'{:4d}'.format(iter_train),
                  "\nLoss "+name_total[n]+ " : ",'{:.9f}'.format(loss),
                  "\nTraining Accuracy : ",'{:.9f}'.format(acc_train_iter))
                acc_train += acc_train_iter
                loss_train += loss
                tb_train.add_scalar('loss_final'+name_total[n],loss_final,cnt)
                tb_train.add_scalar('loss_make'+name_total[n],loss_make,cnt)
                tb_train.add_scalar('loss_type'+name_total[n],loss_type,cnt)
                tb_train.add_scalar('loss_'+name_total[n],loss,cnt)
                # print(batch_name)
            # Mean accuraccy of training 
            acc_train /= (iter_train+1)
            loss_train /= (iter_train+1)
            tb_train.add_scalar("Accuracy_"+name_total[n],acc_train,e)
            
            print("\nModel :", name_total[n],
              "\nEpoch :",'{:4d}'.format(e),
              "\nLoss"+name_total[n]+" : ",'{:.9f}'.format(loss_train),
              "\nTraining Accuracy : ",'{:.9f}'.format(acc_train))
            save_name = path_checkpoint+"/"+name_total[n]+"model_"+str(e)+".pkl"
            if isinstance( model , nn.DataParallel):
                torch.save(model.module.state_dict(),save_name)
            else:
                torch.save(model.state_dict(),save_name)
                
            # Testing accuracy
            model.eval()   
            acc_test = 0
            # path_checkpoint = "./"+name_total[n]
            # model.load_state_dict(torch.load(path_checkpoint+"/"+name_total[n]+"model_"+str(e)+".pkl"))
            for iter_test , (_ , batch_x_test , batch_y_test,_,_) in enumerate(loader_eval):    
                gt_x_test   = batch_x_test.float().cuda()
                gt_y_test   = batch_y_test.long()  
                pred_y_test,_,_ = model(gt_x_test)
                # accuracy
                pred_y_test = pred_y_test.cpu()
                acc = compute_accuracy(pred_y_test, gt_y_test.data.numpy())
                acc_test += acc
            # Mean accuraccy & loss of  testing
            acc_test /= (iter_test+1)
            print("\nModel :", name_total[n],
              "\nTesting Accuracy : ",'{:.9f}'.format(acc_test))
            tb_test.add_scalar("Aaccuracy_"+name_total[n],acc_test,e)
    # eval mode
    if mode == "test":
        model_total  ,optimizer_total , tb_total,epoch_total,name_total
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu") 
        model_total[n].to(device).eval()  
        device_id = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        if len(device_id) > 1 :
            model_total[n] = DataParallel(model_total[n])
        loader_test = DataLoader(
            # Torch TensorDataset format    
            dataset = data_test,
            # Mini batch size
            batch_size =1,
            # training : random shuffle
            num_workers = 2
            )
        path_checkpoint = "./"+name_total[n]
        # path_checkpoint = "./"+name
        try :
            # model_total[n].module.load_state_dict(torch.load(path_checkpoint+"/"+name_total[n]+"model_5.pkl"))
            model_total[n].module.load_state_dict(torch.load(ckpt))
        except:
            model_total[n].load_state_dict(torch.load(ckpt))
        try :
            model = model.module
        except:
            model = model
        output = ['id','label']
        
        with open('./output_'+str(epoch_total[n]-1)+'.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output)
        with torch.no_grad():
            for iter_test , (batch_name, batch_x , batch_y,_,_) in enumerate(loader_test):    
                
                gt_x   = batch_x.float().cuda()
                # gt_y   = batch_y.long()  
                try:
                    pred_y,_,_ = model_total[n].module(gt_x)
                except:
                    pred_y,_,_ = model_total[n](gt_x)   
                pred_y = pred_y.cpu()
                pred_label = np.argmax(F.softmax(pred_y.view(-1) , dim = -1).data.numpy())
                pred_name = label_dic[pred_label]
                
                # cross entropy loss
                # acc = compute_accuracy(pred_y, gt_y.data.numpy())
                # acc_test += acc
                with open('./output_'+str(epoch_total[n]-1)+'.csv', 'a', newline='') as csvfile:
                   writer = csv.writer(csvfile)
                   writer.writerow([int(batch_name[0]), pred_name])               
            # Mean accuraccy & loss of  testing
            # acc_test /= (iter_test+1)
            # loss_test /= (iter_test+1)
                print("ID :", int(batch_name[0]) , " Label : ",pred_name)
            print("\nModel :", name_total[n])
            # compute_confusion_matrix(model_total[n], loader_test , name_total[n])

if __name__ == "__main__":
    mode = input("Current mode: ")
    n    = int(input("Net: "))
    if mode == "train":
        main("train",n)
    elif mode == "test":
        ckpt = input("weight: ")
        main("test",n)
        
    else:
        print("Invalid mode")
