#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:45:45 2021

@author: elisson
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def save_images(images, images_path, directory):
    for i in range(len(images)):
        path = images_path[i].replace('Database', directory).split('/')[:-1]

        path = '/'.join(path)
        
        os.makedirs(path, exist_ok=True)       
        
        path += '/' + str(i)+'.png'        
        print(path)
        
        #cv2.imwrite(path, images[i])

def plot_histogram(features, image, image_class, name, channel, image_path, directory):
    names = ['Frios', 'Intermediários', 'Quentes']
    channels = ['Red', 'Green', 'Blue']
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))  
    
    path = image_path.replace('database', directory).replace('Originals', 'Images').split('/')[:-1]
    path.insert(3, channels[channel])
    
    path = '/'.join(path)
    
    os.makedirs(path, exist_ok=True)   
    
    path += '/' + str(name)+'.png'     
    
    cv2.imwrite(path, image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ax[0].imshow(image)
    ax[1].bar(names, features)
    
    print(': ',features)
    
    path = path.replace('Images', 'Histograms').split('/')[:-1]
    path = '/'.join(path)
    
    os.makedirs(path, exist_ok=True) 
    path += '/' + str(name)+'.png'    
    
    print(path)
    
    #ax[1].spines['bottom'].set_color('#ffffff')
    #ax[1].spines['top'].set_color('#ffffff') 
    #ax[1].spines['right'].set_color('#ffffff')
    #ax[1].spines['left'].set_color('#ffffff')
	
    #ax[1].tick_params(axis='x', colors='#ffffff')
    #ax[1].tick_params(axis='y', colors='#ffffff')
	
    #ax[1].yaxis.label.set_color('#ffffff')
    #ax[1].xaxis.label.set_color('#ffffff')
    
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)

    fig.savefig(path, transparent=False, dpi=200) 

def quantize_image(image, n_colors):
    a = []
    for i in image:
        a.append(max(i))
        
    r = int(max(a) // n_colors) + 1
    image = np.uint8(image / r) * r
        
    return image

def processing(image, channel, n):     
    new_image = cv2.split(image)[channel]
    new_image = quantize_image(new_image, n)
          
    return new_image
   
def plot_hist(img):
    histr = cv2.calcHist([img],[0],None,[4],[0,256])
    plt.plot(histr)
    
def exists(aux, n):
    for i in aux:
        if(i == n):
            return True
    return False

def create_hist_base(image, n):
    base = []
    h, w = image.shape
    for i in range(h):                    
        for j in range(w):   
            if(len(base) >= n - 1):
                return base
            
            if(image[i][j] != 0 and not exists(base, image[i][j])):
                base.append(image[i][j])
    return base
    
def find_index(n, base_hist):
    for i in range(len(base_hist)):
        if(n == base_hist[i]):
            return i
    
def calculate_hist(image, base_hist):
    hist = np.zeros(len(base_hist), dtype=np.int32)
    
    h, w = image.shape
    
    for i in range(h):            
        for j in range(w): 
            if(image[i][j] != 0):
                index = find_index(image[i][j], base_hist)
                hist[index] += 1
    
    return hist

def extract_thermal_hist(images, images_path, n, image_class):    
    data = []
        
    for i in range(len(images)):    
        for j in range(1, 2):
            new_image = processing(images[i], j, n)
            
            if(i == 0):
                base_hist = create_hist_base(new_image, n)
                                                
            hist = calculate_hist(new_image, base_hist)              

            #plot_histogram(hist, new_image, image_class, i, j, images_path[i], 'hist')
                   
        data.append(hist)
                                        
    return data
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    