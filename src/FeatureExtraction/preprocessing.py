#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:40:45 2021

@author: elisson
"""

import cv2
import os
import glob
import numpy as np

def load_images_with_cv2(images_path):
    print('Carregando imagens...\n')
    images = []
    for image in images_path:
        img = cv2.imread(image)
        images.append(img)
        
    return images

def class_names(path):
    class_names = []
    for i in range(len(path)):
        a = path[i].split('/')
        class_names.insert(i, a[len(a) - 1]) 
    
    return class_names

def paths():
    paths = []
    paths.append(glob.glob('../../database/Doente/*'))
    paths.append(glob.glob('../../database/Saudavel/*'))
  
    return paths  
        
def resize(images):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (256, 256))
    
    return images
        
def segmentate_images(images, masks):    
    for i in range(len(images)):
        rows, columns, _ = images[i].shape
   
        for y in range(rows):
                for x in range(columns):                    
                    if(np.all(masks[i][y][x]) == 0):
                        images[i][y][x] = [0, 0, 0]
    
    return images

def filter_images(images, images_path):
    for i in range(len(images)):
        img = cv2.GaussianBlur(images[i], (13, 13), 0)
                
        img = images[i]        
        cv2.imwrite(images_path[i], img)

def save_images(images, images_path, directory):
    for i in range(len(images)):
        path = images_path[i].replace('database', directory).split('/')[:-1]

        path = '/'.join(path)
        
        os.makedirs(path, exist_ok=True)       
        
        path += '/' + str(i)+'.png'        
        print(path)
        
        cv2.imwrite(path, images[i])
                
if(__name__ == "__main__"):   
    paths = paths()
          
    for i in range(len(paths)):
        print('----------------------------------------------------------------\n', paths[i], '\n----------------------------------------------------------------\n')        
           
        images_path = sorted(glob.glob('{}/*'.format(paths[i][0])))   
        masks_path = sorted(glob.glob('{}/*'.format(paths[i][1])))   
        
        images = load_images_with_cv2(images_path)
        masks = load_images_with_cv2(masks_path)
        
        images = resize(images)
        masks = resize(masks)

        images = segmentate_images(images, masks)
        
        save_images(images, images_path, 'Database')
        save_images(masks, masks_path, 'Database')
        
        
        
        
        

    
    
        
    
    
