from img_preprocess import *
from img_feature import *
from img_cluster import *
from img_show_cluster import *

# for large images: do the following first before calling the function

# from PIL import Image
# Image.MAX_IMAGE_PIXELS = None


def mix_n_match_cluster(path_to_library, n_init = 25, number_of_clusters = 5):

    image_ready, valid_path = img_ready(path_to_library)
    features, feature_list = img_get_feature(image_ready, valid_path)
    scaled_feature_list = scale_feature(feature_list)
    result_dict = KMean_cluster(valid_path, scaled_feature_list, feature_list, n_init = n_init, number_of_clusters = number_of_clusters)
    show_result_in_plot(result_dict)
    print_results(result_dict)



