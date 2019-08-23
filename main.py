# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:18:01 2019

@author: BurakBey
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
import os
import alexmodel
from exemplary_dataload import  GraspDataset,Rescale,ToTensor
import copy
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from matplotlib.path import Path
import math



class Geomath():
    def __init__(self,labels=None):
        self.labels = labels
        # x,y,h,w,theta 
        # x,y - center coordinates
        #h,w -  its size
        # theta - orientations
    def createPolygon(self,item):
        # x,y,h,w,theta 
        # x,y - center coordinates
        #h,w -  its size
        # theta - orientations
                
        x,y,h,w,t = item
        coords = [(-w, -h), (w, -h), (w, h), (-w, h)]
        p = Polygon(coords)
        #print(x,y,h,w,t)
        return translate(rotate(p,t), x,y)
    
    def findMaskImage(self,item):
        x,y,h,w,t = item
        coords = [(-w, -h), (w, -h), (w, h), (-w, h)]
        p = Polygon(coords)
        
        p = translate(rotate(p,t), x,y)
        p.exterior.coords.xy
        corners = []
        count= 0
        for x,y in p.exterior.coords:
           
           corners.append((x,y))
           count += 1
           if(count>= 4):
               break
           
        nx, ny = 224, 224
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        
        points = np.vstack((x,y)).T
        
        path = Path(corners)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))
        
        return (grid)
    def findIOU(self,item1,item2):
        # intersection over union
        poly1 = self.createPolygon(item1)
        poly2 = self.createPolygon(item2)
        intersect = poly1.intersection(poly2).area
        print('Int = ' , intersect)
        un = poly1.union(poly2).area
        print('Un = ' ,un)
        if(un <= 0):
            return 0
        return intersect/un
    def findDistance(self,item1,item2):
        return( math.pow((item1[0] - item2[0]),2)  +  math.pow((item1[1] - item2[1]),2))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this statement. Each of these
    #   variables is model specific.
    model_ft = None
 
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

    return model_ft, input_size


class BurakLoss(nn.Module):
    def __init__(self):
        self.mesh = torch.meshgrid(torch.linspace(0,224,steps = 2) , torch.linspace(0,224,steps = 2))
        # arrange this mesh such that it will be in range of h,w respectively, for each item in batch
        super(BurakLoss, self).__init__()
    def forward(self, outputs, labels):
        pX, pY = self.trans_params(outputs)
        gtX, gtY = self.trans_params(labels) # gt = ground truth, p = predicted
        
        #return nn.functional.mse_loss(pX,gtX)  + nn.functional.mse_loss(pY,gtY)
        
        x = outputs[:,0]
        y = outputs[:,1]
        h = outputs[:,2]
        w = outputs[:,3]
        theta = outputs[:,4]
        
        lx = labels[:,0]
        ly = labels[:,1]
        lh = labels[:,2]
        lw = labels[:,3]
        ltheta = labels[:,4]
        
        tX = x - 0.5
        tY  = y - 0.5
        sX = w/224
        sY = h/224
        
        ltX = lx - 0.5
        ltY  = ly - 0.5
        lsX = lw/224
        lsY = lh/224
        
        #12.555 is used, due to the error caused by the normalization

        firstRow = torch.stack([sX* torch.cos(theta/12.555), -sX* torch.sin(theta/12.555), tX])

        secondRow = torch.stack([sY* torch.sin(theta/12.555), sY* torch.cos(theta/12.555), tY])
        
        
        lfirstRow = torch.stack([lsX* torch.cos(ltheta/12.555), -lsX* torch.sin(ltheta/12.555), ltX])

        lsecondRow = torch.stack([lsY* torch.sin(ltheta/12.555), lsY* torch.cos(ltheta/12.555), ltY])
        
        
       
        firstRow = torch.transpose(firstRow,0,1)
        secondRow = torch.transpose(secondRow,0,1)
        
        lfirstRow = torch.transpose(lfirstRow,0,1)
        lsecondRow = torch.transpose(lsecondRow,0,1)
        
        a = torch.tensor([[  0., 224.],[  0., 224.],[  1.,   1.]])
        initialTransformed = torch.stack([self.mesh[0].flatten() , self.mesh[1].flatten(), torch.ones_like(self.mesh[0].flatten())])
        
        transformedX  = torch.matmul(firstRow,a)
        transformedY  = torch.matmul(secondRow,a)
        
        ltransformedX  = torch.matmul(lfirstRow,a)
        ltransformedY  = torch.matmul(lsecondRow,a)
        
        #print((transformedX))
        intersect = ((torch.min(transformedX[0,1], ltransformedX[0,1]) - torch.max(transformedX[0,0], ltransformedX[0,0]) + 1 ) * (torch.min(transformedY[0,1], ltransformedY[0,1]) - torch.max(transformedY[0,0], ltransformedY[0,0]) + 1 ))
        union = (transformedX[0,0] - transformedX[0,1]) * (transformedY[0,0] - transformedY[0,1])
        
        intersect = torch.abs(intersect)
        union = torch.abs(union)
        print(intersect)
        print(union)
        return 1 - (intersect/union)
        #return nn.functional.mse_loss(transformedX,ltransformedX)  + nn.functional.mse_loss(transformedY,ltransformedY)

    def trans_params(self,params):

        x = params[:,0]
        y = params[:,1]
        h = params[:,2]
        w = params[:,3]
        theta = params[:,4]
        
        tX = x - 0.5
        tY  = y - 0.5
        sX = w/224
        sY = h/224
        #12.555 is used, due to the error caused by the normalization

        firstRow = torch.stack([sX* torch.cos(theta/12.555), -sX* torch.sin(theta/12.555), tX])

        secondRow = torch.stack([sY* torch.sin(theta/12.555), sY* torch.cos(theta/12.555), tY])
        
       
        firstRow = torch.transpose(firstRow,0,1)
        secondRow = torch.transpose(secondRow,0,1)
        
        
        initialTransformed = torch.stack([self.mesh[0].flatten() , self.mesh[1].flatten(), torch.ones_like(self.mesh[0].flatten())])
        
        transformedX  = torch.matmul(firstRow,initialTransformed)
        transformedY  = torch.matmul(secondRow,initialTransformed)
        
        # B X N
        
        return transformedX, transformedY


def plt_grasps(sample_batch, outs):
    showIm = sample_batched['image'][0].numpy() * 224
    showGrasp = sample_batched['grasp'][0][0]
    out = np.zeros((1,5), dtype= 'float64')
    for i in range(5):
        out[0,i] = outs[0,i].detach()
    #x,y,h,w,theta = showGrasp
    x,y,h,w,theta = out[0]
    
    x = x
    y = y
    h = h
    w = w
    showIm = showIm.astype(np.uint8).transpose(1,2,0)
        
    plt.imshow(showIm)
    plt.scatter(  np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  ,np.sin(theta)*w/2 + np.cos(theta)*h/2 +y)
    plt.scatter(  np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  , np.sin(theta)*w/2 - np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  , - np.sin(theta)*w/2+ np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  ,-np.sin(theta)*w/2 - np.cos(theta)*h/2 +y )
    plt.show()
    return 0
    

def measureClose(out,label):
    g = Geomath()

    if( label[0]!= label[0] ):
        return 99999990
    else:
        #return g.findIOU(out,label)
        return g.findDistance(out,label)
        
    

def calculateClosest(outs,labels):
    
    
    lab2 = np.array(labels[0])
    out = np.zeros((1,5), dtype= 'float64')
    for i in range(5):
        out[0,i] = outs[0,i].detach()
    
    ind = 0
    clos_ind = 0
    clos_met = 99999990
    
    for l in lab2:
        if(l[0] <= 0.1 and l[1] <= 0.1 and l[2]<=0.1 ):
            break
        calc  =measureClose(out[0],l)  
        if(calc < clos_met):                        
            clos_met = calc
            clos_ind = ind
        ind +=1

    return clos_ind
    

if __name__=='__main__':
    
    # Change the train data directory here., Apply other transforms if neccessery (defined at exemplary_dataload.py)
    
    #Dataload class is overloaded, ['image'] has the data where ['grasp'] have the labels
    #grasp_dataset = GraspDataset(csv_file='data/deneme_data2.csv',root_dir='/',  transform = transforms.Compose([Rescale(224),ToTensor()]))
    #dataloaders = DataLoader(grasp_dataset, batch_size=1,shuffle=True, num_workers=0)
    grasp_dataset = GraspDataset(csv_file='data/smalldata.csv',root_dir='/',  transform = transforms.Compose([Rescale(224),ToTensor()]))
    
    dataloaders = DataLoader(grasp_dataset, batch_size=1,shuffle=False, num_workers=0)
    
    '''
    For plotting the labels into the image
    
    showIm = ((dataloaders.dataset[3]['image'])).numpy() * 224
    showIm = showIm.astype(np.uint8).transpose(1,2,0)
    showGrasp = dataloaders.dataset[3]['grasp'].squeeze()
    x,y,w,h,theta = showGrasp
    plt.imshow(showIm)
    plt.scatter(  np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  ,np.sin(theta)*w/2 + np.cos(theta)*h/2 +y)
    plt.scatter(x,y)
    plt.scatter(  np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  , np.sin(theta)*w/2 - np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 - np.sin(theta)*h/2 +x  , - np.sin(theta)*w/2+ np.cos(theta)*h/2 +y)
    plt.scatter(  -np.cos(theta)*w/2 + np.sin(theta)*h/2 +x  ,-np.sin(theta)*w/2 - np.cos(theta)*h/2 +y )
    plt.show()
'''

    # Object for geometric operations, measurements, IOU etc.
    #calc = Geomath()
    
    num_classes = 5
    num_epochs = 1
    feature_extract= True
    input_size = 224
    
    
    # VGG16 can be used as well
    model_ft = models.resnet34(pretrained=True)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    #alexnet_modal = models.alexnet(pretrained=True)
    #set_parameter_requires_grad(alexnet_modal, feature_extract)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_ft
    
    
    # Detect if we have a GPU available
    #model = model.to(device)
    
    params_to_update = model.parameters()
    
    
    # For printing out the model parameters, needed for development phase only.
    if(feature_extract):
        params_to_update = []
        for name,param in model.named_parameters():
            if(param.requires_grad):
                params_to_update.append(param)
                print('\t', name)
    else:
        for name,param in model.named_parameters():
            if(param.requires_grad):
                print('\t', name)
                
    
    # Change the loss function here
    
    criterion = nn.MSELoss()
    #criterion = BurakLoss()
    
    #Optimizer object, apply different learning rates to create better learning accuracy
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    #Since we are not using a model with inception, define it always as False
    is_inception = False
    # Train and evaluate
    
    since = time.time()
    
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        counter = 0
        # Each epoch has a training and validation phase
       
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        cntr = 0 
        for i_batch, sample_batched in enumerate(dataloaders):
            cntr += 1
            inputs = sample_batched['image'].to(dtype=torch.float)#.to(device=device)
            
            labels = sample_batched['grasp'].to(dtype=torch.float)#.to(device=device)
            
                
            optimizer_ft.zero_grad()

            with torch.set_grad_enabled(True):
                
                if is_inception:
                    # Again, not used in our case
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:

                    outputs = model(inputs)
                    
                    
                    ind = calculateClosest(outputs,labels)
                    print('Selected gt:')
                    print(labels[:,ind])
                    print('Output:')
                    print(outputs)
                    #loss = criterion(outputs[:,:2], labels[:,0,:2])
                    loss = criterion(outputs[:,:2], labels[:,ind,:2])
                    
                    #print(loss)
                    
                preds = outputs

                loss.backward()
                optimizer_ft.step()
                
                #iou = calc.findIOU(outputs[0],labels[0,0])
                
                if(counter % 5  == 0):
                    
                    
                    
                    '''
                    print(epoch)
                    print(outputs)
                    print(labels)
                    print('*')
                    '''
                    #print(iou)
                    
                    plt_grasps(sample_batched,outputs)
                    
                
                
                
                counter += 2
                

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print(epoch_loss)
        print(epoch_acc)

        # deep copy the model
        
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
   
''' 
    # TEST CODE 
    # take a look at the weights
    for param in model.parameters():
        print(param.data)
    
    for i_batch, sample_batched in enumerate(dataloaders):            
        images = sample_batched['image'].to(dtype=torch.float)#.to(device=device)
        outputs = model(images)
        print(outputs)
'''
    #new_model, hist = train_model(new_model, dataloader, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=False)
    
    
torch.save(model, 'weights.pth')