
from os.path import isfile, join
from os import listdir
import cv2

import numpy as np
import os
import math
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


from imutils import build_montages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

            montages = build_montages(images_plot, (300,300), (6,3))
    
        for montage in montages:
            plt.figure(figsize=(10,10))
            imgplot = plt.imshow(montage)
            plt.show() 

