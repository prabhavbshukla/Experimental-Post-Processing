# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:42:36 2022

@author: prabh
"""
# Necessary libraries import
import numpy as np
from PIL import Image
import os
from scipy import stats
from matplotlib import pyplot as plt
from skimage.feature import canny
from os import listdir
from os.path import isfile, join, splitext

#%% Mutual Information
# To ensure repeatability between the experiments, the camera setting must
# be the same. This is achieved through mutual information of the reference
# and new image taken while beginning the experiment at a different time.

##### The file structure for the following program is as follows #####
# The first two lines imports images from the folder.
# The images are correspondingly converted to greyscale.
# They are converted to numpy array in order to truncate the images and focus on centre.
# Image array are converted back to image format.
# Conditional intensity is determined by subtracting two images pixel by pixel. 
# A historgarm providing stats of intensities is determined.
# Using the histogram data, entropy (image information) is calculated for the - 
# - reference image and for conditional image. 
# Finally a scalar quantity of mutual information is acheived by subtraction of entropy.

def mutual_information(original_image, verification_image): 
    original_image = Image.open(original_image)
    verify_image = Image.open(verification_image)
    
    original_image_gray = original_image.convert('L') 
    original_image_gray = np.array(original_image_gray)
    original_image_gray = original_image_gray[1000:1400, 1000:2200]
    original_image_gray = Image.fromarray(original_image_gray)
   
    verify_image_gray = verify_image.convert('L')
    verify_image_gray = np.array(verify_image_gray)
    verify_image_gray = verify_image_gray[1000:1400, 1000:2200]
    verify_image_gray = Image.fromarray(verify_image_gray)
    
    Conditional_intensity = np.float32(original_image_gray) - np.float32(verify_image_gray)
      
    Conditional_intensity = np.where(Conditional_intensity < 0, 0, Conditional_intensity)
    
    histogram_original_gray, bin_edges_gray = np.histogram(original_image_gray, bins=256)
    histogram_conditional, bin_edges_conditional = np.histogram(Conditional_intensity, bins=256)
    
    prob_occurrence_gray_origin = histogram_original_gray/sum(histogram_original_gray)
    prob_conditional = histogram_conditional/sum(histogram_conditional)
    
    H_X = stats.entropy(prob_occurrence_gray_origin,qk=None,base=2,axis=0)
    H_XY = stats.entropy(prob_conditional,qk=None,base=2,axis=0)
    
    mutual_info = H_X - H_XY
    print("Original image entropy: ",H_X,"\nConditonal entropy: ",H_XY,"\nMutual Information in %: ",mutual_info*100/H_X)

    return original_image_gray, verify_image_gray, H_X, mutual_info

original_image = 'XXX.png'
verification_image = 'YYY.png'
original_image_gray, verify_image_gray, H_X, mutual_info = mutual_information(original_image, verification_image)
#plt.imshow(original_image)
#plt.show()
plt.imshow(verify_image_gray)
plt.show()


#%% Definition of functions related to tangential distortion correction (Homography)

##### The file structure for the following program is as follows #####
# Upper most file (wrt Heirarchy) is a file named 'Homography'
# This file consists of two files: Original and Transformed.
# There is also a file which contains the source points for each caliberated plane.
# The file 'Original' contains files of experimental data from different dates.
# Each experimental data file has angles which contain 16 pictures and a
# file called 'Recovery' which contains 5 pictures (At plane #9) based on angles.

# Original > Experiment > Angles/Recovery > Pictures (.jpg)

# The file 'Transformed' is the exact same as original. It is just an alternate
# savepath to not lose any information by overwriting.

# Since tangential distortion is heavily dependent on the coefficients of the
# transformation matrix, the work flow is as follows:
# 1. Calibration using known planes    
# 2. Coefficients wriiten to a file
# 3. Function to implement transformation using the above file
# 4. Test performed on calibration planes
# 5. Implementation on the images captured during the experiment


# STEP 1: Calculate the coefficients using 4 pixel coordinates each on
# original and desired image.
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# STEP 2: Write coefficients of all calibration planes into a file
def calibration_planes():
    # Source points are the 4 coordinates on the original image
    source_pts = np.genfromtxt("Homography/source_points.txt", skip_header = 2)    
    # Destination points are the 4 coordinates on the desired image
    dst = [(1639, 2058),
           (3063, 2051),
           (3065, 978),
           (1642, 979)]
    to_write = [] # Array to store all the coefficients (of 15 planes)
    for index in range(16):
        src = [(int(source_pts[index,1]),int(source_pts[index,2])),
               (int(source_pts[index,3]),int(source_pts[index,4])),
               (int(source_pts[index,5]),int(source_pts[index,6])),
               (int(source_pts[index,7]),int(source_pts[index,8]))]
        coeffs = find_coeffs(dst, src)
        to_write.append(coeffs)
    # Operation to write to a file
    np.savetxt("Homography/coefficients.txt", to_write, delimiter=" ")

# STEP 3: Function to use the coefficients and transform an image.
# Parameters of the function:
    # orig_path:    Original image path
    # trnsfrm_path: Transformed image path
    # all_coeffs:   Variable containing the all the coefficients row-wise
    # plane_number: Plane at which laser sheet was placed and image captured
    # rotate_req:   Flag variable to designate whether image should be rotated or not
def project_transform(orig_path, trnsfrm_path, all_coeffs, plane_number, rotate_req):
    coeffs = all_coeffs[plane_number,:]
    original = Image.open(orig_path)
    if rotate_req:
        original = original.rotate(180)
    transformed = original.transform((4928, 3264), Image.PERSPECTIVE, coeffs)
    transformed.save(trnsfrm_path, quality = 100)

# STEP 4: Function to implement the projection on the calibration planes
def test_calibration():
    read_path = "Homography/Calibration_Planes/"
    save_path = "Homography/Corrected_Planes/"
    all_coeffs = np.genfromtxt("Homography/coefficients.txt", delimiter = " ")
    for i in range(16):
        project_transform(read_path+str(i)+".jpg", save_path+str(i)+".jpg", all_coeffs, i, 0)

# STEP 5: Function to implement transformation on all images captured
# during the experiment
def image_correction():
    all_coeffs = np.genfromtxt("Homography/coefficients.txt", delimiter = " ")

    # List of Files for main experiments.
    files = ["Experiment_with_canard", "11_04_2022_Baseline", "Experiments_with_canard2"]    
    for file in files:
        if file == "Experiments_with_canard2":
            i_files = ["25 deg", "30 deg", "35 deg"]
        else:
             i_files = ["15 deg", "20 deg", "25 deg", "30 deg", "35 deg"]
        for i_file in i_files:
            for i in range(16):
                read_path = "Homography/Original/" + file + "/" + i_file + "/" + str(i) + ".jpg"
                save_path = "Homography/Transformed/" + file + "/" + i_file + "/" + str(i) + ".jpg"
                project_transform(read_path, save_path, all_coeffs, i, 1)
    
    # Recovery image correction
    for file in files:
        if file == "Experiments_with_canard2":
            i_files = [25, 30, 35]
        else:
            i_files = [15, 20, 25, 30, 35]
        f_name = file + "/Recovery"
        for i in i_files:
            read_path = "Homography/Original/" + f_name + "/" + str(i) + ".jpg"
            save_path = "Homography/Transformed/" + f_name + "/" + str(i) + ".jpg"
            project_transform(read_path, save_path, all_coeffs, 9, 1)      
 
#%% Calling the functions related to tangential distortion correction

# Execute the following during the first ever run
# calibration_planes()

# Running the test is essential for the first run to ensure that
# homography actually takes place. It is optional after first run.
# test_calibration()
image_correction()


#%% Image post processing for better visualisation of vortices. 
# First, saturate the pixel intensities to 0 and 255 if intensity below or above 50, respectively.
# Second, use canny edge detection to visualise better only the necessary structural details -
# - in the images. 

images = {}
mypath = "Base/25 deg/"
image_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
image_list.sort()
i = 0
for image in image_list:
    if splitext(image)[1] != ".jpg":
        image_list.pop(i)
    i += 1
    
for image in image_list:
    if int(splitext(image)[0]) > 5 and int(splitext(image)[0]) < 12:
        img = Image.open(mypath+image)
        img = img.convert('L')
        
        image_array = np.array(img)
        
        for i in range(len(image_array)): 
            for j in range(len(image_array[0])):
                if image_array[i][j] > 50: 
                    image_array[i][j] = 255
                if image_array[i][j] < 50: 
                    image_array[i][j] = 0
                
        edge_image = canny(image_array, sigma=10)
    
        # im_save = im.fromarray(edge_image)
        # img_name = splitext(image)[0]
        # im_save.save('Canny/30_deg_canard2_canny\canny_image_'+ img_name + '.jpg')
