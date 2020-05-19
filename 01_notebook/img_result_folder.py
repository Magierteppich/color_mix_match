import os
import re
from shutil import copyfile

def get_results_in_folder(folder_name, list_of_neighbors):
    new_path = "../02_results" + "/" + folder_name 
    old_path = "../02_results/00_result"
    os.makedirs(new_path)

    pattern = pattern = ".*\/(.*.jpg)"
    valid_file_list = []
    for path in list_of_neighbors:
        valid_file_list.append(re.findall(pattern, path[0])[0])
    
    for file in valid_file_list: 
        copyfile(old_path + "/" + file, new_path + "/" + file)
    
    print(f"All reommended images are stored in {new_path} ")
