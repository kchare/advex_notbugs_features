'''
Helper functions that run projected gradient descent (PGD) using L-infinity and L2 norms.
'''

import tensorflow as tf
import numpy as np

################
## L-infinity ##
################

# Helper for pgd_linf
@tf.function
def onestep_pgd_linf(model, X, y, epsilon, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, model(X + delta))

    delta = tf.clip_by_value(delta + alpha*tf.sign(tape.gradient(loss, delta)), X-epsilon, X+epsilon)
    
    return delta

# Full run – import this 
def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = tf.zeros_like(X)
    for t in range(num_iter):
        delta = onestep_pgd_linf(model, X, y, epsilon, alpha, delta)
    return delta


###############
###   L2   ####
###############

# Helper
def norm(Z):
    """Compute norms over all but the first dimension"""
    return tf.norm(tf.reshape(Z, (Z.shape[0], -1)), axis=1) #tf.norm(Z[:,:,:,:], axis=(0,-1))

########### ROBUSTIFICATION ##############
def single_pgd_step_robust(model, X, y, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.MeanSquaredError()(y, model(X + delta)) # comparing to robust model representation layer
        
    grad = tape.gradient(loss, delta)
    delta -= alpha*grad / norm(grad) # normalized gradient step
    delta = tf.math.minimum(tf.math.maximum(delta, -X), 1-X) # clip X+delta to [0,1]
    
    return delta, loss

@tf.function
def pgd_l2_robust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = single_pgd_step_robust(model, X, y, alpha, delta)

    # Prints out loss to evaluate if it's actually learning (currently broken)
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta

################# Adversarial Examples ###############
def single_pgd_step_adv(model, X, y, alpha, epsilon, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE # Use no aggregation - will give gradient separtely for each ex.
        )(y, model(X + delta)) # comparing to label for original data point

    grad = tape.gradient(loss, delta)
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    
    
    # Updated 12/6/2021 - Move to more intuitive gradient adjustment scheme
    # found here: https://adversarial-ml-tutorial.org/adversarial_examples/
    # - Note: Prev. implementation followed guide at same link but ran into bugs with NaN values
    # - Note: here we *add* the gradient step because adversarial attacks *maximize* the loss function
    z = delta + alpha * (grad / (normgrad + 1e-10)) # add 1e-10 to prevent div by zero error; tf will not raise, just yield NaNs
    normz = tf.reshape(norm(z), (-1, 1, 1, 1)) # adjust broadcasting
    delta = epsilon * z / (tf.math.maximum(normz, epsilon) + 1e-10) 
  
    return delta, loss

def pgd_l2_adv(model, X, y, alpha, num_iter, epsilon=0, example=False):
    """Applies L2 PGD for an adversarial example
    
    Will run `num_iter` iterations of PGD over the examples for L2 ball
    constrained by `epsilon` and step size of `alpha`. 
    
    To optimize performance, will decorate only the interior function. Moreover,
    we will re-instantiate this every time. O/w TF will produce errors related to retracing
    of the computational graph."""

    # Apply tf.function to create computational graph of the single step
    # for optimal performance
    fn = tf.function(single_pgd_step_adv)
    delta = tf.zeros_like(X)
    loss = 0
    for t in range(num_iter):
        delta, loss = fn(model, X, y, alpha, epsilon, delta)
        
    if example:
        print(f'{num_iter} iterations, final CCE {loss}')
    return delta