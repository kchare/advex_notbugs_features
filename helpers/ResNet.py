'''Implements ResNet9,..56 dynamically for CIFAR-10
Description of implementation can be found here: https://arxiv.org/pdf/1512.03385.pdf'''
import tensorflow as tf

class ResNetBlock(tf.keras.layers.Layer):
  '''See official RStudio/Keras documentation here:
  https://github.com/rstudio/keras/blob/main/vignettes/examples/cifar10_resnet.py
  for implemetation of residual block layers
  
  Implements residual block described for CIFAR 10 in
  He et al. (2016): https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
  '''
  def __init__(self, n_filters, kernel_size, stride, init_stride=False, first_layer=False):
    self.n_filters = n_filters
    self.first_layer = first_layer
    super(ResNetBlock, self).__init__()
    if init_stride:
      stride1 = stride + 1
    else:
      stride1 = stride

    self.conv_layer_1 = tf.keras.layers.Conv2D(n_filters, kernel_size, strides=stride1, padding='same', 
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                               kernel_initializer='he_normal')
    self.conv_layer_2 = tf.keras.layers.Conv2D(n_filters, kernel_size, strides=stride, padding='same', 
                                               kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
                                               kernel_initializer='he_normal')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.act1 = tf.keras.layers.ReLU()
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.act2 = tf.keras.layers.ReLU()
    self.conv_projection = tf.keras.layers.Conv2D(n_filters, 1, strides=stride1, padding='same', 
                                                  #kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                  kernel_initializer='he_normal')
  
  def call(self, inputs):
    x = self.conv_layer_1(inputs) # apply without activation since will batch normalize
    x = self.bn1(x)
    x = self.act1(x) # use ReLU activation as specified by paper
    x = self.conv_layer_2(x)
    x = self.bn2(x)
    if self.first_layer:
      inputs = self.conv_projection(inputs)
    x = tf.keras.layers.Add()([x, inputs])
    x = self.act2(x)
    return x


class ResNet56(tf.keras.Model):
  def __init__(self, block_depth, base_filters=16):
    self.block_depth = block_depth

    super(ResNet56, self).__init__()

    self.conv_1 = tf.keras.layers.Conv2D(base_filters, 3, padding='same')
    self.pre_bn = tf.keras.layers.BatchNormalization()

    self.stack1 = [ResNetBlock(base_filters, 3, 1) for _ in range(self.block_depth-1)]
    self.one_to_two = ResNetBlock(base_filters * 2, 3, 1, init_stride=True, first_layer=True)
    self.stack2 = [ResNetBlock(base_filters * 2, 3, 1) for _ in range(self.block_depth - 1)]
    self.two_to_three = ResNetBlock(base_filters * 4, 3, 1, init_stride=True, first_layer=True)
    self.stack3 = [ResNetBlock(base_filters * 4, 3, 1) for _ in range(self.block_depth - 1)]
    self.out_dense =  tf.keras.layers.Dense(10, activation='softmax')
    
  def call(self, inputs):
    x = self.conv_1(inputs)
    x = self.pre_bn(x)
    x = tf.keras.layers.Activation('relu')(x)
    for i in range(self.block_depth-1):
      x = self.stack1[i](x)
    x = self.one_to_two(x)
    for i in range(self.block_depth-1):
      x = self.stack2[i](x)
    x = self.two_to_three(x)
    for i in range(self.block_depth-1):
      x = self.stack3[i](x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.out_dense(x)
    return x
  
  def summary(self):
    """See hack here: https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
    overrides default 'multiple' output shape for debugging, something that is still an open issue on GitHub for TF2.7"""
    x = tf.keras.layers.Input(shape=(32,32,3))
    m = tf.keras.Model(inputs=x, outputs=self.call(x))
    return m.summary()
