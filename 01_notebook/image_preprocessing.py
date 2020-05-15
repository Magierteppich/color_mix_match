
from os.path import isfile, join
from os import listdir
import cv2


def get_file_path(path_to_library):
	    
	file_list = [path_to_library + "/" + f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
	    
	return file_list

def img_read(file_list):
    
    img_list = []
    valid_path = []        

    for file_path in file_list: 
  
    # read all images and put the image-array into the img_list
        
        img = cv2.imread(file_path) # CV2 reads in BGR! 
    
        if img is None: 
            print(f"File {file_path} is not readable.")  

        else:
            img_list.append(img)
            valid_path.append([file_path])
        
    return img_list, valid_path 

	
	
def img_resize(img_list, height = 220, width = 220):
    
    dim = (width, height)
    list_resize = []
    
    for i in range(len(img_list)):
        res = cv2.resize(img_list[i], dim, interpolation = cv2.INTER_LINEAR)
        list_resize.append(res)
    
    return list_resize
	
	
def img_denoise(list_resize):
    
    list_denoise = []
    for i in range(len(list_resize)):
        blur = cv2.GaussianBlur(list_resize[i], (5,5), 0)
        list_denoise.append(blur)

    return list_denoise

	
def img_ready(path_to_library, height = 220, width = 220): 

	file_list = get_file_path(path_to_library)
	img_list, valid_path = img_read(file_list)
	list_resize = img_resize(img_list, height = height, width = width)
	list_denoise = img_denoise(list_resize) 
	img_ready = list_denoise.copy()

	return img_ready
