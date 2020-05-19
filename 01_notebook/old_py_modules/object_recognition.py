# the module loads the trained model to predict input images 
# the modell is trained on 10 categories 

from os.path import isfile, join
from os import listdir
import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import preprocessing
from  tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img


def image_processing (file_path): #returns image into 32x32 np.array 

    input_image = image.load_img(file_path, target_size = (32, 32))
    input_image = image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis = 0)
    
    return input_image


def load_model (model_path, model_weight_path): #load the object recognition model (10 categories)
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weight_path)
    
    return loaded_model

def predicted_object (input_image, loaded_model, category_dict): #apply the model on the converted input image and returns a prediction // category_dict translates the number to object

    results = loaded_model.predict(input_image)
    results_index = np.where(results.max())
    index_max = np.argmax(results)
    category_list = list(category_dict)
    predicted_object = category_list[index_max]
    
    return predicted_object

def prediction (file_path, model_path, model_weight_path, category_dict): # it combineds all 3 functions above and return the prediction for 1 input. 
    
    return predicted_object (image_processing(file_path), load_model (model_path, model_weight_path), category_dict)


def predicted_object_df (path_to_library): # extend prediction function to go thru all files in a folder
    
    files = [f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] # read all files in the path_to_library file and put them into a list
    predicted_object_list = []

    for f in files: # use the prediction function, go thru all files and put the predictions into a list
        predicted_object_list.append(prediction (path_to_library + "/" + f, model_path, model_weight_path, category_dict))

    prediction_df = pd.DataFrame(zip(files, predicted_object_list)) # return a df with each picture (file path) and the predicted object
    
    return prediction_df
