from img_preprocess import *
from img_feature import *
from img_knn import *
from img_show_knn import *

import os
import re
from shutil import copyfile

def mix_n_match_knn(path_to_library, target_image, number_of_neighbors, height = 220, width = 220):

    image_ready, valid_path = img_ready(path_to_library, height, width)
    img_ready_gray = color_to_gray(image_ready)

    features, feature_list = img_get_feature(image_ready, valid_path)
    target_index = find_target_image(valid_path, target_image)

    if target_index is None: 
        return 
    else:
        list_img_mse = get_img_mse(img_ready_gray, target_index)
        list_img_ssim = get_img_mse(img_ready_gray, target_index)
        feature_knn, features_knn = get_feature_list_knn(valid_path, features, feature_list, list_img_mse, list_img_ssim)
        scaled_feature_list_knn = scale_feature(feature_knn)
        list_of_neighbors = find_neighbors(valid_path, features_knn, scaled_feature_list_knn, number_of_neighbors, target_index)
        show_result_in_plot_knn(list_of_neighbors)
        print_results(list_of_neighbors, target_index, number_of_neighbors, valid_path)

