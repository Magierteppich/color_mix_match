from img_preprocess import * 
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import image as img
from collections import Counter
from matplotlib import pyplot as plt

def img_dominant_color(image_ready, k=3):
    
    img_dominant_color = []
    
    for img in image_ready: 
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB to get the right order
        img = img.reshape((img.shape[0] * img.shape[1], 3))
    
        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(img)
        
        #count labels to find most popular
        label_counts = Counter(labels)
        
        
        dominant_color = [] #for each image, the list of all k dominant colors. [(r,g,b), (r,g,b), (r,g,b)]
        
        for cluster_center in clt.cluster_centers_:
            b, g, r = cluster_center
            dominant_color.append((
                    b / 255,
                    g / 255,
                    r / 255
                  ))
        plt.imshow([dominant_color])
        plt.show()
        
        
        img_dominant_color.append(dominant_color)
            
    return img_dominant_color 

#result is a list of sub-lists. Each sub-list contains all k dominant colors for each image. 
# [[(r,g,b), (r,g,b), (r,g,b)],[(r,g,b), (r,g,b), (r,g,b)]] - img_dominant_color for 2 images with respectively 2 dominant colors each.