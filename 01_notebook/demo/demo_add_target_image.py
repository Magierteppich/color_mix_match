# from load_demo import *
# import numpy as np 
# from demo_img_preprocess import *
# from demo_img_feature import *
# from demo_img_knn import *

def pre_process_target_image(target_image, target_image_path, target_feature_list, feature_list, valid_path, image_ready):

    all_in_feature_list = target_feature_list + feature_list
    all_in_valid_path = target_image_path + valid_path 
    all_in_image = target_image + image_ready

    return all_in_valid_path, all_in_image, all_in_feature_list



