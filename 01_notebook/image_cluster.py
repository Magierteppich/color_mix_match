
from os.path import isfile, join
from os import listdir
import cv2

import numpy as np
import os
import math
from sklearn.cluster import KMeans
from collections import Counter

from sklearn.preprocessing import StandardScaler


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


def img_get_feature(path_to_library, height = 220, width = 220, k=4): # returns a list of dictionary containing ALL image features.

    file_list = get_file_path(path_to_library)
    preprocessed_img = img_ready(path_to_library, height = height, width = width)

    img_list, valid_path = img_read(file_list)
    list_hsv = img_hsv(img_ready = preprocessed_img)
    list_colorfulness = img_colorfulness(img_ready = preprocessed_img)
    list_contrast = img_contrast(img_ready = preprocessed_img)
    list_dominant_color = img_dominant_color(img_ready = preprocessed_img, k=k)

    feature_list = []
    features = ["H", "S", "V", "colorfulness", "contrast", "R", "G", "B"]
    for i in range(len(valid_path)):
        temp = list_hsv[i] + list_colorfulness[i] + list_contrast[i] + list_dominant_color[i]
        feature_list.append(temp)


    return valid_path, features, feature_list 

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# Scaling the features: 


def scale_feature(feature_list):
    
    scaler = StandardScaler()
    scaled_fit = scaler.fit(feature_list)
    scaled_feature_list = scaled_fit.transform(feature_list)
    
    return scaled_feature_list #2d array of scaled features. The order of the value is the same as valid_path and features (from the img_get_feature fucntion)


def KMean_cluster (valid_path, scaled_feature_list, feature_list): 
    
    if len(scaled_feature_list) <= 10:
        number_of_clusters = 2
    elif 10 < len(scaled_feature_list) < 50:
        number_of_clusters = 5
    elif 20 < len(scaled_feature_list) < 200:
        number_of_clusters = 12
    else:
        number_of_clusters = 20
    
    # KMeans model 
    model = KMeans(n_clusters = number_of_clusters, n_init = 10, init = "random")
    clusters = model.fit(scaled_feature_list)
    KMean_cluster = clusters.predict(scaled_feature_list)
    
    # KMean cluster result / combine image path with features
    KMean_cluster_list = list(KMean_cluster) # list of results from the clustering. 
    KMean_result = np.concatenate((valid_path, scaled_feature_list, feature_list), axis = 1) # list of images with corresponding features 

    # put all images belonging to 1 cluster into 1 list (temp) and join all the temps into a cluster_image list. 
    cluster_image = []
    cluster = 0
    while cluster <= number_of_clusters:
        index = [i for i in range(len(KMean_cluster_list)) if KMean_cluster_list[i] == cluster] # get the position of all elements, which belong to a cluster
        temp = KMean_result[tuple(index),:] # filter the KMean_result by index to get the images with their features
        cluster_image.append(temp) 
        cluster = cluster + 1
    
    #returning a dictionary with the results
    clusters_iter = list(range(number_of_clusters))
    result_dict = {}
    for cluster in clusters_iter:
        result_dict[f"cluster {cluster + 1}"] = list(cluster_image[cluster][:,0])
    # print(f"cluster {x}: {result[x][:,0]}")
    
    return result_dict #dictionary{cluster x : image names beloing to it}

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# Showing results: 

def show_result_dict_path (path_to_library, height = 220, width = 220):

    valid_path, features, feature_list = img_get_feature(path_to_library, height = height, width = width, k=4)
    scaled_feature_list = scale_feature(feature_list)
    result_dict = KMean_cluster(valid_path, scaled_feature_list, feature_list)

    return result_dict 
    

def img_resize_plot(img, height = 220, width = 220): # it takes a image (as array) and resize it. 
    
    dim = (width, height)
    list_resize = []
    
    img_res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    
    return img_res


