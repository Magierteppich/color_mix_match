
import numpy as np
import math
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import colour
from img_preprocess import *

# Here, only feature extraction functions are stored. 
# img_read is a list of images (as np.array) - from img_preprocess import * needed 

 
def img_hsv(image_ready):

    '''
    The function takes in a list of images.
    For each image, it returns a list containing the average h, s, and v value. 
    The function returns a list of sub-lists. Each sub-list represents an image and consists of 3 values (avg_h, avg_s, avg_v)
    '''
    
    img_hsv = []
    
    for img in image_ready:
        
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



def img_colorfulness(image_ready):

    '''
    The function takes in a list of images.
    For each image, it returns a list containing the c-metric.
    The function returns a list of sub-lists. Each sub-list represents an image and consists of 1 value, the c-metric. 
    '''
    
    img_colorfulness = []
    
    for img in image_ready: 
        
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
    
    return img_colorfulness 


def img_contrast(image_ready):

    '''
    The function takes in a list of images.
    For each image, it returns a list containing the contrast.
    The function returns a list of sub-lists. Each sub-list represents an image and consists of 1 value, the contrast. 
    '''
    
    img_contrast = []
    
    for img in image_ready: 
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
        contrast = img.std()
            
        temp_result = list([contrast])
        img_contrast.append(temp_result)
    
    return img_contrast 


def img_dominant_color(image_ready, k=4):

    '''
    The function takes in a list of images.
    For each image, it applies the KMeans method to identify the cetroids. Number of cetroids can be defined by k.
    And the center of the centroid with the most dots belonging to it, is the dominant color. 
    The function returns a list of sub-lists. Each sub-list represents an image and consists of 3 values, R,G,B value of the dominant color. 
    '''

    img_dominant_color = []
    
    for img in image_ready: 
    
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
            
    return img_dominant_color 


def average_RGB(image_ready):
    
    '''
    Takes a list of images and calculate the average RGB for each image.
    '''

    average_RGB = []
    for img in image_ready: 
        (B, G, R) = cv2.split(img.astype("float"))    
        temp = [np.average(R), np.average(G), np.average(B)]
        average_RGB.append(temp)
        
    return average_RGB

def convert_RGB_to_kelvin (average_RGB):

    '''
    Takes the list of average_RGBs and convert each RGB to a kelvin value using the hernandez1999 method. 
    The function returns a list of sub-lists. Each sublist contains 1 kelvin value. 
    '''
    
    img_kelvin = []
    
    for image in average_RGB: 
        
        #Assuming sRGB encoded colour values.
        RGB = np.array(image)

        # Conversion to tristimulus values.
        XYZ = colour.sRGB_to_XYZ(RGB / 255)

        # Conversion to chromaticity coordinates.
        xy = colour.XYZ_to_xy(XYZ)

        # Conversion to correlated colour temperature in K.
        CCT = [colour.xy_to_CCT(xy, 'hernandez1999')]
        
        img_kelvin.append(CCT)
    
    return img_kelvin   

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# putting all features into 1 feature list. 

def img_get_feature(image_ready, valid_path, height = 220, width = 220, k=4): # returns a list of dictionary containing ALL image features.
    
    '''
    The function combines all feature generating steps described above.
    It takes the list of images and their valid pathes and return the feature_list. 
    The feature_list is a list of sublists. Each sublist contains values corresponding to the features as described in "features".
    Features is just a list of features. 
    '''
    
    list_hsv = img_hsv(image_ready)
    list_colorfulness = img_colorfulness(image_ready)
    list_contrast = img_contrast(image_ready)
    list_dominant_color = img_dominant_color(image_ready, k=k)
    list_average_RGB = average_RGB(image_ready)
    list_kelvin = convert_RGB_to_kelvin(list_average_RGB)

    feature_list = []
    features = ["H", "S", "V", "colorfulness", "contrast", "R", "G", "B", "kelvin"]
    for i in range(len(valid_path)):
        temp = list_hsv[i] + list_colorfulness[i] + list_contrast[i] + list_dominant_color[i] + list_kelvin[i]
        feature_list.append(temp)


    return features, feature_list 
