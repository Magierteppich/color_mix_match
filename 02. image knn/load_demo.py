import pickle

def load_demo(file_name):
    infile = open(file_name, "rb")
    img_ready, valid_path, features, feature_list = pickle.load(infile)
    infile.close()
    
    return img_ready, valid_path, features, feature_list