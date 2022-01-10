#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:57:15 2021

@author: elisson
"""

import cv2
import numpy as np
import glob
import SimpleITK as sitk
import pandas as pd

import matplotlib.pyplot as plt

from mpi4py import MPI

import texture_features as tf
import radiomics_features as rf
import thermal_hist as th

import logging

logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

def load_images_with_simpleITK(images_path, name):
    print('Carregando {} com simpleITK...\n'.format(name))
    images = []
    for image in images_path:
        img = sitk.ReadImage(image)        
        images.append(img)
        
    return images

def load_images_with_cv2(images_path, name):
    print('Carregando {} com cv2...\n'.format(name))
    images = []
    for image in images_path:
        img = cv2.imread(image)
        images.append(img)
    
    # simg = sitk.GetImageFromArray(images[0], isVector=False)
    # plt.imshow(simg)
    # plt.show()
        
    return images

def save_svc(data, name):
    print('{} características estraídas com sucesso!\n\nConsulte o arquivo {} para conferir.'.format(len(data[0]) - 1, name))
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(name, index=False, header=False)    

def paths():
    paths = []
    paths.append(glob.glob('../../Database/Doente/*'))
    paths.append(glob.glob('../../Database/Saudavel/*'))
  
    return paths  

def classification_images(n, image_class):
    classes = []
    for i in range(n):
        aux = []
        aux.append(image_class)
    
        classes.insert(len(classes), aux)
    
    return classes

def separate_directories(myrank):
    images_path = []
    masks_path = []    
    local_masks_path = []
    local_images_path = []    
    classes_process = []
    
    if(myrank == 0):    
        directories = paths()
        for i in range(len(directories)):                   
            images_path.insert(i, sorted(glob.glob('{}/*'.format(directories[i][0]))))   
            masks_path.insert(i, sorted(glob.glob('{}/*'.format(directories[i][1]))))                          
        
        if(nprocs == 1):
            return np.arange(len(directories)), images_path, masks_path
        
        n = int(len(images_path[0])/nprocs * 2) + 1
                
        for i in range(nprocs // len(directories)):
            for j in range(len(directories)):
                classes_process = np.append(classes_process, j)
                local_images_path.insert(i, images_path[j][i * n : (i + 1) * n])
                local_masks_path.insert(i, masks_path[j][i * n : (i + 1) * n])
                
    return classes_process, local_images_path, local_masks_path

def extract_features(image_class, local_images_path, local_masks_path):    
    local_images = load_images_with_simpleITK(local_images_path, 'imagens')
    local_masks = load_images_with_simpleITK(local_masks_path, 'mascaras')  
        
    radiomics = rf.extract_radiomics_features(local_images, local_masks)
                
    local_images = load_images_with_cv2(local_images_path, 'imagens')
    
    terminal_hist = th.extract_thermal_hist(local_images, local_images_path, 4, image_class)    
        
    texture = tf.extract_texture_features(local_images)
    
    classes = classification_images(len(local_images), image_class)
                            
    local_dataset = np.concatenate((radiomics, texture, terminal_hist, classes), axis=1)    
    
    return local_dataset

def execute(nprocs, myrank):                    
    dataset = []    
    dataset = []
        
    classes_process, local_images_path, local_masks_path = separate_directories(myrank)
                
    if(nprocs == 1):
        for i in range(len(local_images_path)):
            dataset.append(extract_features(classes_process[i], local_images_path[i], local_masks_path[i]))
        return dataset
    
    classes_process = comm.bcast(classes_process, root=0)
    
    local_images_path = comm.scatter(local_images_path, root = 0)
    local_masks_path = comm.scatter(local_masks_path, root = 0)
                                                    
    local_dataset = extract_features(classes_process[myrank], local_images_path, local_masks_path)
        
    dataset = comm.gather(local_dataset, root = 0) 
    
    return dataset
               
if(__name__ == "__main__"):      
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    myrank = comm.Get_rank()
    
    data = []

    dataset = execute(nprocs, myrank) 
            
    if(myrank == 0):  
        for d in dataset:
            for f in d:
                data.insert(len(data), f)
        data = np.array(data)
        
        save_svc(data, '../../dataset.csv')
           
    
    
        
    
    
