#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:47:11 2021

@author: elisson
"""
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import six

def extract_radiomics(image, mask):    
    params = { 'force2D': True,
          'force2Dextration': True,
          'force2Ddimension': True}

    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    name = []
    data = []
    selector = sitk.VectorIndexSelectionCastImageFilter()
    selector.SetIndex(1)
    image = selector.Execute(image)
    
    results = extractor.execute(image, mask, 1)
    
    for key, val in six.iteritems(results):             
        data.append(val)
        name.append(key)
                    
    return data[18:], name[18:]

def extract_radiomics_features(images, masks):
    print('\nExtraindo características radiômicas\n')
    data = []
    
    for i in range(len(images)): 
        features = []        
        
        images[i] = sitk.Cast(images[i], sitk.sitkVectorFloat32)
        masks[i] = sitk.Cast(masks[i], sitk.sitkVectorFloat32)
        
        radiomics, _ = extract_radiomics(images[i], masks[i])
        
        features = np.concatenate(radiomics, axis=None) 

        data.append(features)
                    
    return data
