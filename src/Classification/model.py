#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 21:42:46 2021

@author: elisson
"""

import tensorflow as tf


def create_model(units, input_dim, hidden_layers, optimizer='adam'):
    classifier = tf.keras.models.Sequential()
    
    classifier.add(tf.keras.layers.Dense(units=units, activation='relu', input_dim=input_dim))
    classifier.add(tf.keras.layers.Dropout(rate = 0.2))
    
    for i in range(hidden_layers):
        classifier.add(tf.keras.layers.Dense(units=units, activation='relu'))
        classifier.add(tf.keras.layers.Dropout(rate = 0.2))    
    
    classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier