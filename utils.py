import numpy as np
import pandas as pd

label_dir = 'labels.csv'

def convert_to_onehot(label_dir, no_of_features = 120):
    """
    input:
          label_dir = has the image_id, breed_name and the corresponding class label_column. 
          no_of_features = Number of dog breeds
    output:
          returns a numpy array of size (data_size, no_of_features)
    """
    labels_list = pd.read_csv(label_dir)
    data_size = labels_list.shape[0]
    label_column = labels_list['labels']                                #Extracting only the labels column
    label_column = np.asarray(label_column, dtype = np.int32)
    label_column = np.reshape(label_column, (label_column.shape[0], 1))
    one_hot = np.zeros((data_size, no_of_features), dtype = np.int32)
    for i in range(data_size):
        class_value = label_column[i][0]
        index = (class_value - 1)
        one_hot[i][index] = 1
    return one_hot
convert_to_onehot(label_dir)    

