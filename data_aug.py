#THE UTILITY FUNCTIONS FOR resizing and DATA AUGMENTATION TECHINQUES
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import scipy
import matplotlib.image as mpimg

def image_resize(images_file_path, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH):
    Images = [] #TO STORE THE RESIZED IMAGES
    tf.reset_default_graph()
    X = tf.placeholder((None, None, 3), tf.float32)
    #Another function is tf.image.resize_image
    tf_resize_img = tf.images.resize_images_with_crop_or_pad(X,IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, image_file_path in enumerate(images_file_path):
            image = mpimg.imread(image_file_path[:, :, :3] #3 prevents the the function from reading the alpha channel of the image
            resize_img = sess.run(tf_resize_img, feed_dict = {X: image})
            Images.append(resize_img)
            
    return Images    
        
#Function to flip the all images(All the images have same size)       
def flip_images(images_file_path):
    Images = []
    tf.reset_default_graph()
    
    return Images
    
#Function to add noise to the images(All the images have same size)
def images_with_noise(images_file_path):
    Images = []
    
    return Images