'''
Defines helper functions for data processing
'''

import tensorflow as tf

def normalize_img(img, label):
    """Normalizes image to [0,1] for better model training"""
    return img / 255, label

def make_tf_data(raw_ds):
    """Applies image normalization to raw TF Data structure"""
    ds = raw_ds.map(normalize_img)
    return ds

