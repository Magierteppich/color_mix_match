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


#### Image description 

from PIL import Image
from PIL.ImageStat import Stat
import pandas as pd
import numpy as np
import os
import math
from os.path import isfile, join
from os import listdir
 

def image_brightness (path_to_library):
    
    file_list = [f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] # read all files in the path_to_library file and put them into a list
    brightness_dict = {"file_path" : [],
                       "brightness" : []}
    df = pd.DataFrame.from_dict(brightness_dict)
    
    for file_path in file_list:
    
        img = Image.open(path_to_library + "/" + file_path).convert("L")
        stat = Stat(img)
        brightness = stat.rms[0] #RMS = root-mean-square for each band in the image
        
        df = df.append({"file_path" : file_path,
                        "brightness" : brightness}, ignore_index = True)
        
    return df


def calc_hsv (path_to_library):
    
    file_list = [f for f in listdir(path_to_library) if isfile(join(path_to_library, f))]
    
    df = {"file" : [], 
          "average_h" : [],
          "average_s" : [], 
          "average_v" : []}
    
    df = pd.DataFrame.from_dict(df)
    
    for file_path in file_list:
        
        img = cv2.imread(path_to_library + "/" + file_path, 1)
        
        if img is None: 
            print("Error, image could not be loaded properly")
        else: 
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = []
            s = []
            v = []
    
            for line in img_hsv:
                for pixel in line:
                    temp_h, temp_s, temp_v = pixel
                    h.append(temp_h)
                    s.append(temp_s)
                    v.append(temp_v)
            
            average_h = round(sum(h)/len(h),4)
            average_s = round(sum(s)/len(s),4)
            average_v = round(sum(v)/len(v),4)
            
            df = df.append ({"file" : file_path, 
                             "average_h" : average_h,
                             "average_s" : average_s, 
                             "average_v" : average_v}, ignore_index = True)
    return df




def get_dominant_color(path_to_library, k=4):
    
    result = []
    
    file_list = [f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 

    for file_path in file_list: 
        image = cv2.imread(path_to_library + "/" + file_path) # CV2 reads the array in BGR! 
        
        if image is None: 
            print(f"The image {file_path} is not readable.")
        
        else: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB to get the right order
            image = image.reshape((image.shape[0] * image.shape[1], 3))
    
            #cluster and assign labels to the pixels 
            clt = KMeans(n_clusters = k)
            labels = clt.fit_predict(image)
        
            #count labels to find most popular
            label_counts = Counter(labels)
        
            #subset out most popular centroid
            dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
            image_result = list([file_path]) + list(dominant_color)
            result.append(image_result)
            
    return result #result is a list of sub-lists. Each sub-list contains 4 elements: file_path, r,g,b



def get_colorfulness (path_to_library):
    
    result_colorfulness = []
    
    file_list = [f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
    
    for file_path in file_list: 
        image = cv2.imread(path_to_library + "/" + file_path) # CV2 reads the array in BGR! 
        
        if image is None: 
            print(f"The image {file_path} is not readable.")
            
        else: 
            (B, G, R) = cv2.split(image.astype("float"))

            rg = np.absolute(R-G)
            yb = np.absolute(0.5*(R + G) - B)
            
            (rbMean, rbStd) = (np.mean(rg), np.std(rg))
            (ybMean, ybStd) = (np.mean(yb), np.std(yb))
            
            stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
            meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
            c_metric = stdRoot + (0.3 * meanRoot) 
    
            temp_result = list([file_path]) + list([c_metric])
            result_colorfulness.append(temp_result)
    
    return result_colorfulness #result is a list of sub-lists. Each sub-list contains 4 elements: file_path, r,g,b
    