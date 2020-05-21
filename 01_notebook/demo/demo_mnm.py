from load_demo import load_demo
from demo_img_preprocess import img_ready
from demo_img_preprocess import color_to_gray
from demo_img_feature import img_get_feature
from demo_img_knn import find_target_image
from demo_img_knn import get_img_mse
from demo_img_knn import get_img_ssim
from demo_img_knn import get_feature_list_knn
from demo_img_knn import scale_feature
from demo_img_knn import find_neighbors
from demo_img_show_knn import *


def demo_mix_n_match(file_path, target_image_name, pickle_file_name, number_of_neighbors):

    print("LOAD IMAGE DATABASE")
    print("-----------------------------------------------------\n")
    image_ready, valid_path, features, feature_list = load_demo(pickle_file_name)
    print("\n")
    print("PREPROCESS TARGET IMAGE")
    print("-----------------------------------------------------\n")
    target_image, target_image_path = img_ready(file_path)
    target_features, target_feature_list = img_get_feature(target_image, target_image_path)
    all_in_feature_list = target_feature_list + feature_list
    all_in_valid_path = target_image_path + valid_path
    all_in_image = target_image + image_ready
    target_index = find_target_image(all_in_valid_path, target_image_name)

    if target_index is None: 
        return

    else:
        print("IDENTIFY SIMILAR IMAGES")
        print("-----------------------------------------------------\n")
        all_in_image_g = color_to_gray(all_in_image)
        list_img_mse = get_img_mse(all_in_image_g, target_index)
        list_img_ssim = get_img_ssim(all_in_image_g, target_index)
        feature_list_knn, features_knn = get_feature_list_knn(all_in_valid_path, features, all_in_feature_list, list_img_mse, list_img_ssim)
        scaled_feature_list_knn = scale_feature(feature_list_knn)
        list_of_neighbors = find_neighbors(all_in_valid_path, features_knn, scaled_feature_list_knn, number_of_neighbors, target_index)
        show_target_image(list_of_neighbors)
        print_results(list_of_neighbors, target_index, number_of_neighbors, valid_path)
        show_result_in_plot_knn(list_of_neighbors)

