
import cv2
import numpy as np
import os
import math


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim



# defining the target image

def find_target_image(valid_path, target_image):

    '''
    Identify the position of the target image within target_image. 
    As the target image is "manualy" added thru the function "demo_combine_target_database", the target_index suppose to be 0, always. 
    '''

    target_index = None

    for path in valid_path: 
        if (target_image in path[0])==True:
            target_index = (valid_path.index(path))
    
    if target_index is None:
        print(f"The image {target_image} given cannot be found. Please try again.")
        return None 
    else:
        return target_index


# calculate MSE and SSIM as features based on the target_image selected


def get_img_mse(img_ready_gray, target_index):

    '''
    Calculate the MSE between the target image and all other images in the dataset. 
    '''
    
    target_img_g = img_ready_gray[target_index]
    list_img_mse = []

    for img_g in img_ready_gray: 

        err = np.sum(((target_img_g.astype("float") - img_g.astype("float")) ** 2))
        err /= float(target_img_g.shape[0] * img_g.shape[1])
        list_img_mse.append([err]) 
    
    print(f"The comparison between the target image and all images in the set has been completed (MSE).")
    print("-----------------------------------------------------------------------------\n")
    return list_img_mse


def get_img_ssim(img_ready_gray, target_index):

    '''
    Calculate the SSIM between the target image and all other images in the dataset. 
    '''
    
    target_img_g = img_ready_gray[target_index]
    list_img_ssim = []
    for img_g in img_ready_gray: 
        similariy = ssim(target_img_g, img_g)
        list_img_ssim.append([similariy])
        
    print(f"The structural similarity comparison between the target image and all images in the set has been completed (SSIM).")
    print("-----------------------------------------------------------------------------\n")
    return list_img_ssim
    
# add the features to the features_list

def get_feature_list_knn(valid_path, features, feature_list, list_img_mse,list_img_ssim):

    '''
    Extend the feature_list with MSE and SSIM results. 
    '''

    feature_list_knn = []
    features_knn = features + ["MSE"] + ["SSIM"]
    for i in range(len(valid_path)):
        temp = feature_list[i]+ list_img_mse[i] + list_img_ssim[i]
        feature_list_knn.append(temp)
    
    return feature_list_knn, features_knn


# scaling the features

def scale_feature(feature_list_knn):

    '''
    Scale all features. 
    '''
    
    scaler = StandardScaler()
    scaled_fit = scaler.fit(feature_list_knn)
    scaled_feature_list_knn = scaled_fit.transform(feature_list_knn)
    
    print("All values have been scaled.")
    print("-----------------------------------------------------------------------------\n")
    return scaled_feature_list_knn #2d array of scaled features. The order of the value is the same as valid_path and features (from the img_get_feature fucntion)


# find the x neighbors matching to target image

def find_neighbors(valid_path, features_knn, scaled_feature_list_knn, number_of_neighbors, target_index):

    '''
    Apply KNN to find the neighbors. The first result in the list_of_neigbors is the target image itself. 
    '''
    
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
