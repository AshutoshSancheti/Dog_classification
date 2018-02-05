#THE UTILITY FUNCTIONS FOR resizing and DATA AUGMENTATION TECHINQUES
#Thinkk of using transforms from OpenCV
#Rotated and shifted images
# Image cropping methods - Refer the Inception_v1 paper
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
import scipy
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler
import glob
from tensorflow.python.ops import control_flow_ops #for tf.cond()

my_path = "train/*.jpg"
resize_path = "output/"
IMAGE_SIZE_HEIGHT = 300
IMAGE_SIZE_WIDTH = 300

def image_resize_keep_aspect(image, new_shorter_side):
    """
    Input: image from glob
    Returns resized image preserving the aspect ratio
    
    image_path = "your_image.jpg"
    image_string = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image_int = tf.cast(resized_image, tf.uint8)
    image_enc = tf.image.encode_jpeg(image_int)
    fwrite = tf.write_file("my_resized_image.jpeg", image_enc)
    """
    #while using glob height is the first dimension and width is the second while in tf.image.decode_jpeg it's width
    #tf.reset_default_graph() AssertionError: Do not use tf.reset_default_graph() to clear nested graphs. showed
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    new_shorter_side = tf.constant(new_shorter_side, dtype = tf.float64)
    
    def compute_longer_side(initial_larger_side, initial_shorter_side, new_shorter_side):
        new_larger_side = (new_shorter_side / initial_shorter_side) * initial_larger_side
        new_larger_side = tf.constant(new_larger_side, dtype = tf.float64)
        return new_larger_side
        
    height_less_equal_width = tf.less_equal(height, width)  #returns True if height<=width
    
    #new_dims contains the new dimensions of the image [height, width]
    #if pred is true, then true_fn is executed with shorter_side
    new_dims = tf.cond(pred = height_less_equal_width, 
                    true_fn = lambda: [new_shorter_side, compute_longer_side(width, height, new_shorter_side)],
                    false_fn = lambda: [compute_longer_side(height, width, new_shorter_side), new_shorter_side])
    X = tf.placeholder(tf.float32, shape = [None, None, 3])
    tf_resized_image = tf.image.resize_images(X, new_dims)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    resized_image = sess.run(tf_resized_image, feed_dict = {X: image})
    return resized_image
    

#Try a method where they resize the image keeping the same aspect ratio and then crop - Done
def image_resize(images_file_path, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, PATH):
    Images = [] #TO STORE THE RESIZED IMAGES
    images = glob.glob(images_file_path)
    
    tf.reset_default_graph()  
    X = tf.placeholder(tf.float32, shape = [None, None, 3])
    #tf_resize_img = tf.image.resize_images(X, (IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    #tf.image.resize_images compresses or expands the images changing the aspect ratio therefore use the image_resize_keep_aspect() first
    # and then the resize_image_with_crop_or_pad() function. Maybe you can use it separately
    
    tf_resize_img = tf.image.resize_image_with_crop_or_pad(X,IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH)
    #USE BOTH OF THEM FOR IMAGE AUGMENTATION IN SEPARATE FUNCTION
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 1
        #for index, image_file_path in enumerate(images_file_path):
        for image_in in images:
            image = mpimg.imread(image_in)[:, :, :3] #3 used to ignore alpha
            
            #Resizing the image keeping the aspect ratio same and then cropping it using resize_image_with_crop_or_pad
            image = image_resize_keep_aspect(image, new_shorter_side = 300)
            resize_img = sess.run(tf_resize_img, feed_dict = {X: image})
            #Should I convert it into a numpy array?
            scipy.misc.imsave(PATH + str(i) + ".jpg", resize_img)
            Images.append(resize_img)
            i = i + 1
            
    return Images    
image_resize(my_path, 300, 300, resize_path)


#Function to normalize the images in the range 0 to 1
#This function takes in resized input
def normal_image(images):
    scaler = MinMaxScaler(feature_range = (0, 1))
    rescaled_images = scaler.fit_transform(images)
    return images


#Function: Use resized images from image_resize() and then flip     
def flip_images(images_file_path, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, PATH):   
    Flipped_Images = [] #TO STORE THE RESIZED IMAGES
    images = glob.glob(images_file_path)
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = [IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 3])
    flip_image = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for image in images:
            image = mpimg.imread(image_in)[:, :, :3]
            flipped_image = sess.run(flip_image, feed_dict = {X: image})
            #flipped_image = np.array(flipped_image, dtype = np.float32)
            scipy.misc.imsave(PATH + str(i) + ".jpg", flipped_image)
            Flipped_Images.append(flipped_image)
    return Images
    
#Function to add noise to the images(All the images have same size)
def images_with_noise(images_file_path):
    Noise_Images = []
    images = glob.glob(images_file_path)
    
    mean = 0.0
    std_dev = np.sqrt(0.5)
    for image in images:
        gauss_noise = np.random.random((IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH,1), np.float32)
        gauss_noise = np.concatenate((gauss_noise, gauss_noise, gauss_noise), axis = 2)
        gauss_image = cv2.addWeighted(image, 0.75, 0.25 * gaussian, 0.25, 0)
        Noise_Images.append(gauss_image)
    Noise_Images = np.array(Noise_Images, dtype = np.float32)
    return Noise_Images
    

    