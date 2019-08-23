# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:10:50 2019

@author: BurakBey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
      

class Grasp():
        
    def __init__(self,CURPATH):   
        self.metadata = None # folder info
        self.listcontainer  = None # file info in each folder
        self.PATH = CURPATH
        self.RGBCONST = 'RGB.png'
        self.GRASPCONST = '_grasps.txt'
        
        
        self.metadata,self.listcontainer = self.load_metadata(self.PATH)
        self.container  = np.array(self.listcontainer)
        for i in range(self.container.shape[0]):
            self.container[i] = np.array(self.container[i])
            
        #self.pandaObject= self.readTarget2(newFrame)

    
    def getTarget(self,obj,pose,index):
        return self.data[obj][pose][1][index]
    
    def load_metadata(self, path):
        metadata = []
        folder_count = 0
        container = []
        self.folder_names = []

        start = time.time()
        for i in os.listdir(path):
            self.folder_names.append(i)
            
            container.append([])
            for f in os.listdir(os.path.join(path, i)):
                metadata.append(IdentityMetadata(path, i, f))
                if( self.is_it_RGB (f,self.RGBCONST)):
                    container[folder_count].append(self.findNameForImages(f))
            folder_count += 1
        end = time.time()
        print(str(end-start) + 'seconds to load folders')
        return np.array(metadata),container
        
    
 
        
    def readTarget2(self,newFrame,normalized = False):
        start = time.time()
        counter = 0
      
        for obj in self.container:            
            for path in obj:
                newFrame[counter,1] = self.createRGBPath(self.PATH,path)
                df = pd.read_csv(self.createGraspPath(self.PATH,path), sep = ';', header = None)
                df = df.values
                df[:,0:4] = df[:,0:4] / 1024
                df[:,0:4] = df[:,0:4] * 224
                print(df[0])
                for i in range(len(df)):
                    newFrame[counter, i*5 + 2] = df[i,0]
                    newFrame[counter, i*5 + 3] = df[i,1]
                    newFrame[counter, i*5 + 4] = df[i,2]
                    newFrame[counter, i*5 + 5] = df[i,3]
                    newFrame[counter, i*5 + 6] = df[i,4]
                counter += 1
            end = time.time()
            print(str(end-start) + 'seconds to load ' + str(counter) + ' images')
            

        return newFrame
       
        
    def findNameForImages(self,str1):
        # designed for the jacquard dataset representation
        return str(str1).split('_')[0] + '_' + str(str1).split('_')[1] 
    
    def is_it_RGB(self,filename,RGBCONST):
        return(str(filename)[-7:] == self.RGBCONST)
        
    def createGraspName(self,path):
        return path + self.GRASPCONST
    
    def createGraspPath(self,fullpath,folder):
        return fullpath + '/' + folder.split('_')[1] + '/' + self.createGraspName(folder)
    
    def createRGBName(self,path):
        return path + '_' + self.RGBCONST
    
    def createRGBPath(self,fullpath,folder):
        return fullpath + '/' + folder.split('_')[1] + '/' + self.createRGBName(folder)
    
    def image_load(self,path):
        return cv2.resize(cv2.imread(path), (224,224))
        

    def findMaxLabels(self):
        maxLabels = 0
      
        for obj in self.container:            
            for path in obj:
                df = pd.read_csv(self.createGraspPath(self.PATH,path), sep = ';',header=None)
                if(len(df) > maxLabels):
                    maxLabels = len(df)
                    print(maxLabels)
        return maxLabels





g = Grasp('C:/Users/BurakBey/Desktop/Graduation Project/final/data/jacquard')
l = g.findMaxLabels()
p  = np.array(['a'])
p.resize((4706,))
nums  = np.zeros((4706,l*5 +1), dtype='float64')
df =  np.vstack([p,nums.T]).T
df = pd.DataFrame(df)


np_values = g.readTarget2(df.values)
toFrame = pd.DataFrame(np_values)
# I got errors while converting it to csv due to floating points, thus round the float nums to int
toFrame.to_csv('data/yeni.csv', float_format='%g')
np_values[:5,:8]