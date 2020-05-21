
import numpy as np
import math
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import colour
from demo_img_preprocess import *

# Here, only feature extraction functions are stored. 
# img_read is a list of images (as np.array) - from img_preprocess import * needed 

 
def img_hsv(image_ready):
    
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
            
    print("Hue, saturation and brightness have been determined for the target image.")
    return img_hsv



def img_colorfulness(image_ready):
    
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
    
    print("The colorfulness index has been calculated.")
    return img_colorfulness #result is a list of sub-lists. Each sub-list contains 2 elements: file_path, colorfulness (the higher the number, the more colorful)


def img_contrast(image_ready):
    
    img_contrast = []
    
    for img in image_ready: 
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
        contrast = img.std()
            
        temp_result = list([contrast])
        img_contrast.append(temp_result)

    print("The contrast has been calculated.")
    return img_contrast #result is a list of sub-lists. Each sub-list contains 2 elements: file_path, contrast (the higher the number, the higher the contrast


def img_dominant_color(image_ready, k=4):
    
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
            
    print("The dominant color has been determined.")
    return img_dominant_color #result is a list of sub-lists. Each sub-list contains 4 elements: file_path, r,g,b


def average_RGB(image_ready):
    
    average_RGB = []
    for img in image_ready: 
        (B, G, R) = cv2.split(img.astype("float"))    
        temp = [np.average(R), np.average(G), np.average(B)]
        average_RGB.append(temp)
        
    return average_RGB

def convert_RGB_to_kelvin (average_RGB):
    
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
    print("The image temperature has been determined.")
    return img_kelvin   #img_kelvin is a list of calculated Kelvin value (based on average RGB and hernandez1999 method) for each image in img_ready

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# putting all features into 1 feature list. 

def img_get_feature(image_ready, valid_path, height = 220, width = 220, k=4): # returns a list of dictionary containing ALL image features.
    
    # file_list = get_file_path(path_to_library)
    # preprocessed_img = img_ready(path_to_library, height=220, width=200)
    # img_list, valid_path = img_read(file_list)
    
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

    print("All charateristics have been calculated and stored.")
    return features, feature_list 
