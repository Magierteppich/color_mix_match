# color_mix_match

Ironhack Data Anyltics Bootcamp - final project 

### Goal: 

Build an application to sort images based on image features, e.g. dominant color, contrast, structural similarity, colorfulness etc. 

### Approach:

1) Pre-process images:
    - read images from given folder 
    - resize images (mostly shrink the size)
    - denoise the image
    - convert color image to grayscale

2) Calculate image features:
    - calculate HSV
    - Calculate colorfulness
    - Calculate contrast 
    - Calculate dominant color 
    - Calculate temperature

3) Used maschined learning
    - a) Use Kmeans to cluster the images - number of clusters can be defined - called "Image cluster" going forward
    - b) Use Neigherest neighbor to return recommendations based on a target image - called "Image knn" going forward

4) Show results
    - a) Plot clusters and print a the list of all images belonging to each cluster
    - b) Plot the target image, the recommended images and their file pathes. 

### File structure

00. data - containing all test images used
01. mage cluster - module used to cluster a set of images
02. Image knn - based on target image - module used to make recommendations based on target image
03. Notebooks - notebooks used to build and test the modules
04. object recognition - using tensorflow to recognize objects on images / not implemented in the final project
05. dominant color only - module used to return 3 dominant colors per image. 
dominant color only"

#### 01. Image cluster

- image_preprocess: img_preprocess.py
- image_feature: img_feature.py
- image_cluster: img_cluster.py
- img_show_cluster: img_show_cluser.py
- final module: mix_n_match_cluster.py
- image_cluster_demo --> workbook to run the final module 


#### 02.Image knn

To use the module, inputs needed are:
A folder with all images to select from and a separate folder with 1 target image. 

- To generate the *image database* to make recommendations from, run thru the first 2 steps of image cluster.
- **Save** the key variables *resulting features* from the first 2 steps into a **pickle file**: store_demo.py
- **Load** the pickle file: load_demo.py
- **target image preprocess**: demo_img_preprocess.py /_It's the same "img_preprocess.py" as in the image cluster folder to pre-process the target image first. Only difference is the printing of results for demo._
- **target image feature detection**: demo_img_feature.py / _It's the same "img_preprocess.py" as in the image cluster folder to extract the features of the target image._
- **Integrate** the target image results into the loaded database for further process: demo_combine_target_database.py
- **Run KNN** on the result of 6: demo_img_knn.py
- **Show the results**: demo_img_show_knn.py
- final modlue: demo_mix_n_match_knn.py
- workbook used to run the final module: Mix_n_Match demo
- workbook used to store the image features into a pickle file: presentation_pickle_demo_knn


### example results: 

**cluster results**

<img width="478" alt="Bildschirmfoto 2020-05-25 um 18 48 58" src="https://user-images.githubusercontent.com/61271744/82831822-c20d2080-9eb9-11ea-9140-f8c188cf2df6.png">

**target image results**

<img width="599" alt="Bildschirmfoto 2020-05-25 um 18 48 11" src="https://user-images.githubusercontent.com/61271744/82831906-f8e33680-9eb9-11ea-9ad4-3c2133dfb126.png">

<img width="702" alt="Bildschirmfoto 2020-05-25 um 18 48 24" src="https://user-images.githubusercontent.com/61271744/82831911-fbde2700-9eb9-11ea-8abe-252eb1fe2d26.png">


### final presentation
https://docs.google.com/presentation/d/17q32e9GGdifiA6ztwDIRSkFN-02ZPsv_WfFzQ1xaHJA/edit?usp=sharing


### sources used:
- https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
- https://www.dataquest.io/blog/tutorial-colors-image-clustering-python/
- https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/
- https://stackoverflow.com/questions/38876429/how-to-convert-from-rgb-values-to-color-temperature
- https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
- https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html 
- https://towardsdatascience.com/image-pre-processing-c1aec0be3edf 
