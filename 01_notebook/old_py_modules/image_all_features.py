
from os.path import isfile, join
from os import listdir
import cv2

import numpy as np
import os
import math
from sklearn.cluster import KMeans
from collections import Counter

from sklearn.preprocessing import StandardScaler
import colour

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# Image proprocessing: 

def get_file_path(path_to_library):
	    
	file_list = [path_to_library + "/" + f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
	    
	return file_list

def img_read(file_list):
    
    img_list = []
    valid_path = []        

    for file_path in file_list: 
  
    # read all images and put the image-array into the img_list
        
        img = cv2.imread(file_path) # CV2 reads in BGR! 
    
        if img is None: 
            print(f"File {file_path} is not readable.")  

        else:
            img_list.append(img)
            valid_path.append([file_path])
        
    return img_list, valid_path 

	
	
def img_resize(img_list, height = 220, width = 220):
    
    dim = (width, height)
    list_resize = []
    
    for i in range(len(img_list)):
        res = cv2.resize(img_list[i], dim, interpolation = cv2.INTER_LINEAR)
        list_resize.append(res)
    
    return list_resize
	
	
def img_denoise(list_resize):
    
    list_denoise = []
    for i in range(len(list_resize)):
        blur = cv2.GaussianBlur(list_resize[i], (5,5), 0)
        list_denoise.append(blur)

    return list_denoise

	
def img_ready(path_to_library, height = 220, width = 220): 

	file_list = get_file_path(path_to_library)
	img_list, valid_path = img_read(file_list)
	list_resize = img_resize(img_list, height = height, width = width)
	list_denoise = img_denoise(list_resize) 
	img_ready = list_denoise.copy()

	return img_ready

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# Extracting features: 
 
def img_hsv(img_ready):
    
    img_hsv = []
    
    for img in img_ready:
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = []
        s = []
        v = []
    
        for line in hsv:
            for pixel in line:
                temp_h, temp_s, temp_v = pixel
                h.append(temp_h)
                s.append(temp_s)
                v.append(temp_v)
            
        average_h = round(sum(h)/len(h),4)
        average_s = round(sum(s)/len(s),4)
        average_v = round(sum(v)/len(v),4)
        
        hsv_temp = [average_h, average_s, average_v]
        img_hsv.append(hsv_temp)
            
    return img_hsv



def img_colorfulness(img_ready):
    
    img_colorfulness = []
    
    for img in img_ready: 
        
        (B, G, R) = cv2.split(img.astype("float"))

        rg = np.absolute(R - G)
        yb = np.absolute(0.5*(R + G) - B)

        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
            
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        c_metric = stdRoot + (0.3 * meanRoot) 
    
        temp_result = list([c_metric])
        img_colorfulness.append(temp_result)
    
    return img_colorfulness #result is a list of sub-lists. Each sub-list contains 2 elements: file_path, colorfulness (the higher the number, the more colorful)


def img_contrast(img_ready):
    
    img_contrast = []
    
    for img in img_ready: 
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = img.std()
            
        temp_result = list([contrast])
        img_contrast.append(temp_result)
    
    return img_contrast #result is a list of sub-lists. Each sub-list contains 2 elements: file_path, contrast (the higher the number, the higher the contrast


def img_dominant_color(img_ready, k=4):
    
    img_dominant_color = []
    
    for img in img_ready: 
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB to get the right order
        img = img.reshape((img.shape[0] * img.shape[1], 3))
    
        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(img)
        
        #count labels to find most popular
        label_counts = Counter(labels)
        
        #subset out most popular centroid
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        temp_result = list(dominant_color)
        img_dominant_color.append(temp_result)
            
    return img_dominant_color #result is a list of sub-lists. Each sub-list contains 4 elements: file_path, r,g,b

def average_RGB(img_ready):
    
    list_average_RGB = []
    for img in img_ready: 
        (B, G, R) = cv2.split(img.astype("float"))    
        temp = [np.average(R), np.average(G), np.average(B)]
        list_average_RGB.append(temp)
        
    return list_average_RGB

def convert_RGB_to_kelvin (list_average_RGB):
    
    list_kelvin = []
    
    for image in list_average_RGB: 
        
        #Assuming sRGB encoded colour values.
        RGB = np.array(image)

        # Conversion to tristimulus values.
        XYZ = colour.sRGB_to_XYZ(RGB / 255)

        # Conversion to chromaticity coordinates.
        xy = colour.XYZ_to_xy(XYZ)

        # Conversion to correlated colour temperature in K.
        CCT = colour.xy_to_CCT(xy, 'hernandez1999')
        
        list_kelvin.append([CCT])
    
    return list_kelvin   #img_kelvin is a list of calculated Kelvin value (based on average RGB and hernandez1999 method) for each image in img_ready


# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# putting all features into 1 feature list. 

def img_get_feature(path_to_library, height = 220, width = 220, k=4): # returns a list of dictionary containing ALL image features.

    file_list = get_file_path(path_to_library)
    preprocessed_img = img_ready(path_to_library, height = height, width = width)

    img_list, valid_path = img_read(file_list)
    list_hsv = img_hsv(img_ready = preprocessed_img)
    list_colorfulness = img_colorfulness(img_ready = preprocessed_img)
    list_contrast = img_contrast(img_ready = preprocessed_img)
    list_dominant_color = img_dominant_color(img_ready = preprocessed_img, k=k)
    list_average_RGB = average_RGB(img_ready = preprocessed_img)
    list_kelvin = convert_RGB_to_kelvin (list_average_RGB)

    feature_list = []
    features = ["H", "S", "V", "colorfulness", "contrast", "R", "G", "B", "kelvin"]
    for i in range(len(valid_path)):
        temp = list_hsv[i] + list_colorfulness[i] + list_contrast[i] + list_dominant_color[i] + list_kelvin[i]
        feature_list.append(temp)


    return valid_path, features, feature_list 


    


