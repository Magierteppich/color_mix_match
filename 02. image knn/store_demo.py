import pickle

def pickle_image_feature_set (image_ready, valid_path, features, feature_list, outfile_name = "demo_knn.dat"):

    ''' 
    The function store image_ready, valid_path, features and feature_list as results from img_preprocess and img_feature into a pickle file. 
    '''
    
    file_name_outfile = outfile_name
    for_pickle = [image_ready, valid_path, features, feature_list]
    outfile = open(file_name_outfile, "wb") 
    pickle.dump(for_pickle, outfile)
    outfile.close()