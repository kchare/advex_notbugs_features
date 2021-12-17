'''
Defines helper functions for data processing
'''

import tensorflow as tf

def train_preprocess(img, label):
  img /= 255
  img = tf.cast(img, tf.float32)
  img = tf.image.resize_with_pad(img, 32+4, 32+4)
  img = tf.image.random_crop(img, size=[32, 32, 3])
  img = tf.image.stateless_random_flip_left_right(img, (15, 13))
  return img, label

def test_preprocess(img, label):
  img = tf.cast(img, tf.float32)
  img /= 255
  return img, label
