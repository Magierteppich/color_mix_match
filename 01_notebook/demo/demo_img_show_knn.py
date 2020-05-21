
import cv2

import numpy as np
import os
import math

from imutils import build_montages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from demo_img_preprocess import *
from demo_img_feature import *
from demo_img_knn import * 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
   
# show results

def img_resize_plot(img, height = 220, width = 220): # it takes a image (as array) and resize it. 
    
    dim = (width, height)
    list_resize = []
    
    img_res = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    
    return img_res

def show_result_in_plot_knn(list_of_neighbors):

        images_plot = []
        
        for path in list_of_neighbors:
            img = mpimg.imread(path[0])
            img_res = img_resize_plot(img)
            images_plot.append(img_res)

            montages = build_montages(images_plot, (300,300), (3,2))
    
        for montage in montages:
            plt.figure(figsize=(10,10))
            imgplot = plt.imshow(montage)
            plt.show() 

def print_results(list_of_neighbors, target_index, number_of_neighbors, valid_path):
        target = valid_path[target_index]
        print(f"For the image chosen {target[0]}, the following {number_of_neighbors} images may fit well:")
        print("-----------------------------------------------------")
        list_of_neighbors_print = list_of_neighbors[1:]
        for path in list_of_neighbors_print:
            print(f"{path[0]}")