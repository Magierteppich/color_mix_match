
from os.path import isfile, join
from os import listdir
import cv2

import numpy as np
import os
import math
from sklearn.cluster import KMeans
from collections import Counter

from sklearn.preprocessing import StandardScaler
from imutils import build_montages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from img_preprocess import *
from img_feature import *

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# Scaling the features: 

def scale_feature(feature_list):
    
    scaler = StandardScaler()
    scaled_fit = scaler.fit(feature_list)
    scaled_feature_list = scaled_fit.transform(feature_list)
    
    return scaled_feature_list #2d array of scaled features. The order of the value is the same as valid_path and features (from the img_get_feature fucntion)


def KMean_cluster(valid_path, scaled_feature_list, feature_list, n_init = 25, number_of_clusters = 5): 
    
    # KMeans model 
    model = KMeans(n_clusters = number_of_clusters, n_init = n_init, init = "random")
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