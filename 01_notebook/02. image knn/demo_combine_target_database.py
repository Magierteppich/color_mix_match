def combine_target_database (target_feature_list, feature_list, target_image_path, valid_path, target_image, image_ready):
    
    '''
    The function combines the target image and its features generated into the image database.
    all_in_feature_list, all_in_valid_path, all_in_image contains both the images from the database and the target image.
    '''

    all_in_feature_list = target_feature_list + feature_list
    all_in_valid_path = target_image_path + valid_path
    all_in_image = target_image + image_ready
    return all_in_feature_list, all_in_valid_path, all_in_image