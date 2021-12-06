'''
Defines helper functions for data processing
'''

import tensorflow as tf

def train_preprocess(img, label):
    """Mapping function for TF Data pipeline structure"""
    # Bring image to [0,1] scale and cast as float
    img /= 255
    img = tf.cast(img, tf.float32)
    
    # At train time, perform data augmentation
    # Ilyas et al. (2019) record doing data aug. but do not
    # specify how. Fall back to He et a. (2015) def'n
    img = tf.image.resize_with_pad(img, 32+4, 32+4) # Add 4 pixels of zeros
    img = tf.image.random_crop(img, size=[32, 32, 3]) # randomly crop back to size
    img = tf.image.stateless_random_flip_left_right(img, (15, 13)) # randomly horizontally flip
    return img, label

def test_preprocess(img, label):
    """Implements test set preprocessing, which does not include data augmentation"""
    img = tf.cast(img, tf.float32)
    img /= 255
    return img, label