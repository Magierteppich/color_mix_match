
import cv2
import numpy as np
import os
import math


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from img_preprocess import *
from img_feature import *
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# defining the target image 

def find_target_image(valid_path, target_image = "target_01"):
    target_index = 0
    for path in valid_path: 
        if (target_image in path[0])==True:
            target_index = (valid_path.index(path))
        
    return target_index


# calculate MSE and SSIM as features based on the target_image selected


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
