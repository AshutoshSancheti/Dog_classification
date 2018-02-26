#Have a utils.py to contain the graph and the weights, the loss
# Another to standardize and data aug: cropping, resizing, etc.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import inception_blocks_v4
import glob
import matplotlib.image as mpimg
import os
from tensorflow import data
from utils import *
slim = tf.contrib.slim

#Pretrained_weights = "/checkpoint_files/inception_v4.ckpt"
ckpt_dir = "checkpoint_files/inception_v4.ckpt"
new_ckpt_dir = "/checkpoint_files/model.ckpt"
image_size = 299
image_dir = "/output/*.jpg"
labels_dir = "labels.csv"

num_classes = 1001
iterations = 100

#convert image data to glob
image_in = glob.glob(image_dir)

#convert into onehot encoding
targets = convert_to_onehot(labels_dir, no_of_features = 1001)
#assert statement
assert targets.shape == (10222, 1001)

#Load batches of input data
def load_batch(batch_size, image = image_in, label = targets, height = image_size, width = image_size):
    #Should I convert image and label to tensors
    #image = mpimg.imread(image_in)[:, :, :, :3]
    #assert image.shape == (10222, 299, 299, 3), 'Image shape not good'
    images, labels = tf.train.batch([image, label], batch_size = batch_size, num_threads = 1, capacity = 2 * batch_size, enqueue_many=True)
    assert images.shape == (batch_size, 299, 299, 3), 'Batch image not correct' 
    images = tf.cast(images, dtype = tf.float32)
    return images, labels



                                    #######################################################
                                    #                   TRAINING BLOCK                    #
                                    #######################################################
       
#Change the num_classes in inception_blocks_v4.py     
# scope = InceptionV4
with tf.device('/device:GPU:0'):
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        with slim.arg_scope(inception_blocks_v4.inception_v4_arg_scope()):
            X_input = tf.placeholder(tf.float32, shape = [None, image_size, image_size, 3])
            Y_label = tf.placeholder(tf.float32, shape = [None, num_classes])
            
            try:
                pretrained_weights = slim.assign_from_checkpoint_fn(ckpt_dir, slim.get_model_variables('InceptionV4'))
                with tf.Session() as sess:
                    pretrained_weights(sess)
            except ValueError:
                print("The checkpoint file has some error")
                
            logits, end_points = inception_blocks_v4.inception_v4(inputs = X_input, num_classes = num_classes, is_training = True, create_aux_logits= False)
            
            
            predictions = tf.nn.softmax(logits)
            loss = tf.losses.mean_squared_error(labels = Y_label, predictions = predictions)
            
            # Add the loss function to the graph.   
            #loss = tf.losses.softmax_cross_entropy(onehot_labels = Y_label, logits = my_layer_logits)  
            #onehot_labels are the actual labels and logits are the predictions, we don't need to take the softmax of labels
            # The total loss is the user's loss plus any regularization losses. slim.losses.get_total_loss() is deprecated
            # to calc regularization losses use tf.losses.get_regularization_losses
            total_loss = tf.losses.get_total_loss()

            # Specify the optimizer and create the train op:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = slim.learning.create_train_op(total_loss, optimizer) 
            
            #Generating batch
            images, labels = load_batch(32)
            print(images.shape)
            print(labels.shape)
            # Run the training inside a session.
            final_loss = slim.learning.train(train_op,logdir = new_ckpt_dir, number_of_steps = iterations, save_summaries_secs=5,log_every_n_steps=50)(feed_dict = {X_input:images , Y_label: labels})
            
print ("Finished training. Last batch loss:", final_loss)
print ("Checkpoint saved in %s" % new_ckpt_dir)
        
    
                           