
from os.path import isfile, join
from os import listdir
import cv2

import numpy as np
import os
import math
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


from imutils import build_montages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# This section is specific to knn_model: 

# defining the target image 

def find_target_image(valid_path, target_image = "target_01"):
    target_index = 0
    for path in valid_path: 
        if (target_image in path[0])==True:
            target_index = (valid_path.index(path))
        
    return target_index


# calculate MSE and SSIM as features based on the target_image selected

def color_to_gray(img_ready):
    
    img_ready_gray = []
    for img in img_ready: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_ready_gray.append(gray)

    return img_ready_gray

def get_img_mse(img_ready_gray, target_index):
    
    target_img_g = img_ready_gray[target_index]
    list_img_mse = []

    for img_g in img_ready_gray: 

        err = np.sum(((target_img_g.astype("float") - img_g.astype("float")) ** 2))
        err /= float(target_img_g.shape[0] * img_g.shape[1])
        list_img_mse.append([err]) 

    return list_img_mse

def get_img_ssim(img_ready_gray, target_index):
    
    target_img_g = img_ready_gray[target_index]
    list_img_ssim = []
    for img_g in img_ready_gray: 
        similariy = ssim(target_img_g, img_g)
        list_img_ssim.append([similariy])
        
    return list_img_ssim

# add the features to the features_list

def get_feature_list_knn(valid_path, features, feature_list, list_img_mse,list_img_ssim):
    feature_list_knn = []
    features_knn = features + ["MSE"] + ["SSIM"]
    for i in range(len(valid_path)):
        temp = feature_list[i]+ list_img_mse[i] + list_img_ssim[i]
        feature_list_knn.append(temp)
    
    return feature_list_knn, features_knn

# scaling the features

def scale_feature(feature_list_knn):
    
    scaler = StandardScaler()
    scaled_fit = scaler.fit(feature_list_knn)
    scaled_feature_list_knn = scaled_fit.transform(feature_list_knn)
    
    return scaled_feature_list_knn #2d array of scaled features. The order of the value is the same as valid_path and features (from the img_get_feature fucntion)


# find the x neighbors matching to target image

def find_neighbors(valid_path, features_knn, scaled_feature_list_knn, number_of_neighbors, target_index):
    
    model_knn = NearestNeighbors(metric= "cosine",
                                 algorithm = "brute",
                                 n_jobs = -1)
    model_knn.fit(scaled_feature_list_knn)
    target = np.array(scaled_feature_list_knn[target_index])
    score, neighbor_index = model_knn.kneighbors(target.reshape(1, -1), n_neighbors=number_of_neighbors+1)
    neighbor_index = list(neighbor_index[0])
    
    list_of_neighbors = []
    for neighbor in neighbor_index:
        list_of_neighbors.append(valid_path[neighbor])
        
    return list_of_neighbors
   
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# show results

def img_resize_plot(img, height = 220, width = 220): # it takes a image (as array) and resize it. 
    
    dim = (width, height)
    list_resize = []
    
    img_res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    
    return img_res

def show_result_in_plot_knn(list_of_neighbors):

        images_plot = []
        
        for path in list_of_neighbors:
            img = mpimg.imread(path[0])
            img_res = img_resize_plot(img)
            images_plot.append(img_res)

            montages = build_montages(images_plot, (300,300), (6,3))
    
        for montage in montages:
            plt.figure(figsize=(10,10))
            imgplot = plt.imshow(montage)
            plt.show() 

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# combining everything together

def mix_n_match_neighbors(path_to_library, target_image, number_of_neighbors, height = 220, width = 220):

    preprocessed_img = img_ready(path_to_library, height, width)
    img_ready_gray = color_to_gray(preprocessed_img)
    valid_path, features, feature_list = img_get_feature(path_to_library = path_to_library)
    target_index = find_target_image(valid_path, target_image = target_image)
    list_img_mse = get_img_mse(img_ready_gray, target_index)
    list_img_ssim = get_img_mse(img_ready_gray, target_index)
    feature_knn, features_knn = get_feature_list_knn(valid_path, features, feature_list, list_img_mse, list_img_ssim)
    scaled_feature_list_knn = scale_feature(feature_knn)
    target_index = find_target_image(valid_path, target_image = target_image)
    list_of_neighbors = find_neighbors(valid_path, features_knn, scaled_feature_list_knn, number_of_neighbors, target_index)
    show_result_in_plot_knn(list_of_neighbors)