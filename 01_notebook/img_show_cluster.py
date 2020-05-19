
import cv2
import numpy as np
import math

from imutils import build_montages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from img_preprocess import *
from img_feature import *
from img_cluster import *

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #

# plotting the results 

def img_resize_plot(img, height = 220, width = 220): # it takes a image (as array) and resize it. 
    
    dim = (width, height)
    list_resize = []
    
    img_res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    
    return img_res

def show_result_in_plot(result_dict):
    
    for keys in result_dict.keys():
        result_images_path = result_dict.get(keys)
        print(f"{keys}")

        images_plot = []
        
        for path in result_images_path:
            img = mpimg.imread(path)
            img_res = img_resize_plot(img)
            images_plot.append(img_res)

            montages = build_montages(images_plot, (300,300), (5,3))
    
        for montage in montages:
            plt.figure(figsize=(10,10))
            imgplot = plt.imshow(montage)
            plt.show() 

def print_results(result_dict):
    for keys in result_dict.keys():
        result_img_path = result_dict.get(keys)
        print("-----------------------------------------------")
        print(f"The following images belong to {keys}:")

        for path in result_img_path:
            print(path)
            

