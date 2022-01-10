#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:43:27 2021

@author: elisson
"""

import numpy as np
import mahotas as mt
import cv2

def extract_haralick(image):
  textures = mt.features.haralick(image)
  ht_mean = textures.mean(axis=0)
  return ht_mean

def extract_tas(image):
  textures = mt.features.tas(image)
  return textures

def extract_lbp(image):
    features = mt.features.lbp(image, 4, 8, ignore_zeros=False)
    return features

def extract_zernike_moments(image):
    features = mt.features.zernike_moments(image, 4)
    return features

def gray_scale(images):
    gray_images = []
    for i in range(len(images)):
        gray_images.append(cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY))
    
    return gray_images

def extract_texture_features(images):
    print('\nExtraindo características de textura\n')
    data = []
    features = []
    
    gray_images = gray_scale(images)
    
    for i in range(len(gray_images)):
        #haralick = extract_haralick(images[i])
        tas = extract_tas(gray_images[i])
        lbp = extract_lbp(gray_images[i])
        #zernike = extract_zernike_moments(images[i])
 
        features = np.concatenate((lbp, tas), axis=None)
        data.append(features)
    
    return data
