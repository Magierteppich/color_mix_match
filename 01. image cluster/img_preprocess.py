
from os.path import isfile, join
from os import listdir
import cv2


def get_file_path(path_to_library):

    '''
    Given the path to a library, the function reads all files within the directory 
    and return a list containing all file pathes.
    '''
	    
    file_list = [path_to_library + "/" + f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
	    
    return file_list #simple list with file pathes of all files in the directory ["path1", "path2"...]

def img_read(file_list):

    '''
    Read all images on the file_list and store the image array into the img_list. 
    If single filepath cannot be read, an error message will be raised. However, the function continues.
    Valid_path is a list containing all valid pathes. (file_list = valid_path + unreadable files.)

    '''
    
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
    # img_list is a list of images. Each image is a 3d np. array
    # valid_path is a list of list of valid image pathes. [["path_1"], ["path_2"], ["path_3"]...]

	
	
def img_resize(img_list, height = 220, width = 220):

    '''
    Resize image for easiert process. Height and width can be set individually. 
    '''
    
    dim = (width, height)
    list_resize = []
    
    for i in range(len(img_list)):
        res = cv2.resize(img_list[i], dim, interpolation = cv2.INTER_LINEAR)
        list_resize.append(res)
    
    return list_resize #list of images. Each image is a 3d np.array
	
	
def img_denoise(list_resize):

    '''
    Apply Gaussian Blur to images to reduce noises. 
    '''
    
    list_denoise = []
    for i in range(len(list_resize)):
        blur = cv2.GaussianBlur(list_resize[i], (5,5), 0)
        list_denoise.append(blur)

    return list_denoise #list of images. Each image is a 3d np.array

	
def img_ready(path_to_library, height = 220, width = 220): 

    '''
    Img_ready combines all functions described a above. 
    Input: path to library containing images to be processed. 
    Output: preprocessed images stored in image_ready (list of arrays) and the valid_path. 
    '''

    file_list = get_file_path(path_to_library)
    img_list, valid_path = img_read(file_list)
    list_resize = img_resize(img_list, height = height, width = width)
    list_denoise = img_denoise(list_resize) 
    image_ready = list_denoise.copy()

    return image_ready, valid_path #list of images. Each image is a 3d np.array


def color_to_gray(image_ready):

    '''
    In case gray scale image is needed. It takes a list of images and returns a list of grey scaled images. 
    '''
    
    img_ready_gray = []
    for img in image_ready: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_ready_gray.append(gray)

    return img_ready_gray
