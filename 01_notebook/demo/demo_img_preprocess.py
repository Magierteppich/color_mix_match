
from os.path import isfile, join
from os import listdir
import cv2


def get_file_path(path_to_library):
	    
    file_list = [path_to_library + "/" + f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
	    
    print(f"All files from the folder {path_to_library} have been identified.")
    print("-----------------------------------------------------------------------------\n")
    return file_list #simple list with file pathes of all files in the directory ["path1", "path2"...]


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
        
    print("\n")
    print(f"The target image has been read. The path to the target image is {valid_path}")
    print("-----------------------------------------------------------------------------\n")
    return img_list, valid_path 

	
	
def img_resize(img_list, height = 220, width = 220):
    
    dim = (width, height)
    list_resize = []
    
    for i in range(len(img_list)):
        res = cv2.resize(img_list[i], dim, interpolation = cv2.INTER_LINEAR)
        list_resize.append(res)
    
    print(f"The target image has been resized. It now has a dimension of {height} x {width}.")
    print("-----------------------------------------------------------------------------\n")
    return list_resize #list of images. Each image is a 3d np.array
	
	
def img_denoise(list_resize):
    
    list_denoise = []
    for i in range(len(list_resize)):
        blur = cv2.GaussianBlur(list_resize[i], (5,5), 0)
        list_denoise.append(blur)

    print(f"Noises have been removed from the target image.")
    print("-----------------------------------------------------------------------------\n")
    return list_denoise #list of images. Each image is a 3d np.array

	
def img_ready(path_to_library, height = 220, width = 220): 

    file_list = get_file_path(path_to_library)
    img_list, valid_path = img_read(file_list)
    list_resize = img_resize(img_list, height = height, width = width)
    list_denoise = img_denoise(list_resize) 
    image_ready = list_denoise.copy()

    print(f"The target image has been successfully preprocessed.")
    print("-----------------------------------------------------------------------------\n")
    return image_ready, valid_path  #list of images. Each image is a 3d np.array


def color_to_gray(image_ready):
    
    img_ready_gray = []
    for img in image_ready: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_ready_gray.append(gray)

    return img_ready_gray
