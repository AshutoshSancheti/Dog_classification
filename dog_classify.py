#Have a utils.py to contain the graph and the weights, the loss
# Another to standardize and data aug: cropping, resizing, etc.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import inception_blocks_v4
import glob
import inception.preprocessing
import os


slim = tf.contrib.slim

#Pretrained_weights = "/checkpoint_files/inception_v4.ckpt"
ckpt_dir = "/checkpoint_files/"
image_size = 
image_dir = 

def convert_to_logits():
    





                                    #######################################################
                                    #                   TRAINING BLOCK                    #
                                    #######################################################
       
#Change the num_classes in inception_blocks_v4.py     
# scope = InceptionV4  
with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    with slim.arg_scope(inception_blocks_v4.inception_v4_arg_scope()):
        pretrained_weights = slim.assign_from_checkpoint_fn(os.path.join(ckpt_dir, "inception_v4.ckpt"), slim.get_model_variables('InceptionV4'))
        logits, end_points = inception_blocks_v4.inception_v4(inputs = , num_classes = 120, is_training = True)
        predictions = tf.nn.softmax(logits)
        with tf.Session() as sess:
            pretrained_weights(sess)
        
        # Add the loss function to the graph.   tf.losses.softmax_cross_entropy
        loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
        # The total loss is the user's loss plus any regularization losses.
        total_loss = slim.losses.get_total_loss()

        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer) 

        # Run the training inside a session.
        final_loss = slim.learning.train(train_op,logdir=ckpt_dir, number_of_steps=5000,save_summaries_secs=5,log_every_n_steps=500)
  
print("Finished training. Last batch loss:", final_loss)
print("Checkpoint saved in %s" % ckpt_dir)
        
    
                           