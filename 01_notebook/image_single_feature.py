
import numpy as np
import os
import math
from os.path import isfile, join
from os import listdir
import cv2
import image_preprocessing as ip
from sklearn.cluster import KMeans
from collections import Counter

# Here, only the single feature extraction functions are stored. 
# img_read is a list of images (as np.array)

 
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
    
    average_RGB = []
    for img in img_ready: 
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
        CCT = colour.xy_to_CCT(xy, 'hernandez1999')
        
        img_kelvin.append(CCT)
    
    return img_kelvin   #img_kelvin is a list of calculated Kelvin value (based on average RGB and hernandez1999 method)

