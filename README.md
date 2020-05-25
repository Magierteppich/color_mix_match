# color_mix_match

Ironhack Data Anyltics Bootcamp - final project 

### Goal: 

Sort images based on images features, e.g. dominant color, contrast, structural similarity, colorfulness etc. 

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

### 01. Image cluster

1. image_preprocess: img_preprocess.py
2. image_feature: img_feature.py
3. image_cluster: img_cluster.py
4. img_show_cluster: img_show_cluser.py
5. final module: mix_n_match_cluster.py

### 02.Image knn

To use the module, inputs needed are:
A folder with all images to select from and a separate folder with 1 target image. 

1. To generate the *image database* to make recommendations from, run thru the first 2 steps of image cluster.
2. **Save** the key variables *resulting features* from the first 2 steps into a **pickle file**: 02. image knn/store_demo.py
3. **Load** the pickle file: 02. image knn/load_demo.py
4. **target image preprocess**: 02. image knn/demo_img_preprocess.py 

    _It's the same "img_preprocess.py" as in the image cluster folder to pre-process the target image first. Only difference is the printing of results for demo._

5. **target image feature detection**: 02. image knn/demo_img_feature.py 

    _It's the same "img_preprocess.py" as in the image cluster folder to extract the features of the target image._

6. **Integrate** the target image results into the loaded database for further process: 02. image knn/demo_combine_target_database.py
7. **Run KNN** on the result of 6: 02. image knn/demo_img_knn.py
8. **Show the results**: 01_notebook/02. image knn/demo_img_show_knn.py
9. final modlue:02. image knn/demo_mix_n_match_knn.py

### final presentation
https://docs.google.com/presentation/d/17q32e9GGdifiA6ztwDIRSkFN-02ZPsv_WfFzQ1xaHJA/edit?usp=sharing


### sources used:
https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
https://www.dataquest.io/blog/tutorial-colors-image-clustering-python/
https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/
https://stackoverflow.com/questions/38876429/how-to-convert-from-rgb-values-to-color-temperature
https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html 
https://towardsdatascience.com/image-pre-processing-c1aec0be3edf 