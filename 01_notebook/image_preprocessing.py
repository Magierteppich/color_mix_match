
from os.path import isfile, join
from os import listdir
import cv2


def image_preprocessing(path_to_library): 

	def get_file_path (path_to_library):
	    
	    file_list = [path_to_library + "/" + f for f in listdir(path_to_library) if isfile(join(path_to_library, f))] 
	    
	    return file_list
	
	
	def image_resize (file_list, height = 220, width = 220):
	        
	    for file_path in file_list: 
	        
	        # read all images and put the image-array into the image_list
	        img_list = []
	        img = cv2.imread(file_path) # CV2 reads the array in BGR! 
	            
	        if img is None: 
	            print(f"File {file_path} is not readable.")      
	        else:
	            img_list.append(img)
	    
	    dim = (width, height)
	    res_img = []
	    
	
	    for i in range(len(image_list)):
	        res = cv2.resize(img_list[i], dim, interpolation = cv2.INTER_LINEAR)
	        res_img.append(res)
	    
	    return res_img 
	
	
	def image_denoise (res_img):
	    
	    img_denoise = []
	    for i in range(len(res_img)):
	        blur = cv2.GaussianBlur(res_img[i], (5,5), 0)
	        img_denoise.append(blur)
	        
	    return img_denoise

	img_ready = img_denoise
	
return img_ready
