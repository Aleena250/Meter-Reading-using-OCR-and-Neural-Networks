# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:23:11 2021

@author: EIG
"""

# import the necessary packages
from imutils import contours
import numpy as np
import imutils
import cv2
import pytesseract
from pytesseract import Output
import csv
from PIL import Image
from pytesseract import image_to_string

# import sys
# from sklearn.neighbors import KNeighborsClassifier
# #from tensorflow.keras.models import load_model
#If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# east=r'C:\Users\EIG\frozen_east_text_detection.pb'
import numpy as np
import joblib
import cv2
import os
import matplotlib.pyplot as plt
import re
# load the model
#from keras.models import load_model
#import imutils
#from imutils import contours

#classifier = joblib.load('knn_model.gzip')
#model = joblib.load('NeuralNetwork.gzip')



def auto_canny(image, sigma=0.33):
    # compute the mean of the single channel pixel intensities
    v = np.mean(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
# #-------------------------Reference Image Processing---------------------
# # load the reference OCR-A image from disk, convert it to grayscale,
# # and threshold it, such that the digits appear as *white* on a
# # *black* background
# # and invert it, such that the digits appear as *white* on a *black*
# ref = cv2.imread('template_digits2.jpg')
# ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# ref = cv2.adaptiveThreshold(ref,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY_INV,11,2)
# cv2.imshow('reference',ref)
# cv2.waitKey(0)

# refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
#  	cv2.CHAIN_APPROX_SIMPLE)
# refCnts = imutils.grab_contours(refCnts)
# sorted_contours = contours.sort_contours(refCnts, method="left-to-right")[0]
# print("Number of Contours found in reference image = " + str(len(refCnts))) 
# digits = {}
# cv2.drawContours(ref, refCnts, -1, (0, 255, 0), 3)  
# cv2.imshow('Contours', ref) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

# # loop over the OCR-A reference contours
# for (i, c) in enumerate(sorted_contours):
#  	# compute the bounding box for the digit, extract it, and resize
#  	# it to a fixed size
#  	(x, y, w, h) = cv2.boundingRect(c)
#  	roi = ref[y:y + h, x:x + w]
#  	roi = cv2.resize(roi, (57, 88))
#  	# update the digits dictionary, mapping the digit name to the ROI
#  	digits[i] = roi
     
# # for x in digits:
# #  cv2.imshow('digits_roi',digits[x])
# #  cv2.waitKey(0)
#-------------------------Ends Here--------------------------------------

#-------------------------Actual Image processing------------------------
# load the input image, resize it, and convert it to grayscale
image = cv2.imread('image1.png')
#image_org=image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray, 0,255)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)
#-------------------------Ends Here--------------------------------------


#------------------Find Contour of Digits to crop------------------------
contours_d, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
print("Number of Contours found in actual image= " + str(len(contours_d)))

# Draw all contours
# -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

sorted_contours= sorted(contours_d, key=cv2.contourArea, reverse= True)
x,y,w,h=cv2.boundingRect(sorted_contours[2])
cropped_contour=image[y:y+h, x:x+w]

cv2.imshow('Cropped_Contour_Image',cropped_contour)
cv2.imwrite("Cropped_Contour.png",cropped_contour)
cv2.waitKey(0)
#-------------------------Ends Here---------------------------------------



#-----------------image processing on cropped Image-----------------------

(origH, origW) = cropped_contour.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
scale_percent =200 # percent of original size
width = int(cropped_contour.shape[1] * scale_percent / 100)
height = int(cropped_contour.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(cropped_contour, dim, interpolation = cv2.INTER_CUBIC)
(newW, newH) = resized.shape[:2]
rW = origW / float(newW)
rH = origH / float(newH)
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


#adjusted Image quality using gamma
adjusted = adjust_gamma(resized, gamma=1.5)
cv2.imshow("Adjusted_Images",adjusted)
cv2.waitKey(0)

adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

#threshold_img = cv2.threshold(resized.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
threshold_img = cv2.threshold(adjusted_gray.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('threshold image', threshold_img)
#cv2.imshow('threshold image1', threshold_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Dilating the image
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# Appplying dilation on the threshold image
dilation = cv2.dilate(threshold_img, rect_kernel, iterations = 1)
erosion = cv2.erode(dilation, rect_kernel, iterations=1)
cv2.imshow('dilate image', dilation)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()







# =============================================================================
# # Canny edge detection
# 
# crop_image_edges = cv2.Canny(dilation.copy(),100,100, apertureSize=3, L2gradient=True)
# cv2.imshow('Edged_cropped',crop_image_edges)
# cv2.waitKey(0)
# =============================================================================
# # # edges = cv2.Canny(img, 235, 250)
# plt.imshow(crop_image_edges, cmap='gray')
# #plt.imshow(crop_image_edges1, cmap='gray')

cnts, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_dict = dict()
for cont in cnts:
    x, y, w, h = cv2.boundingRect(cont)
    area = cv2.contourArea(cont)
    if 8< area and 8 < w and h > 20:
        contours_dict[(x, y, w, h)] = cont

print("Number of Contours found in cropped Image= " + str(len(cnts)))
contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)
print("Number of filtered Contours found in cropped Image= " + str(len(contours_filtered)))

resized_contour=resized.copy()
(H, W) = resized_contour.shape[:2]
blank_background = np.zeros_like(dilation)
img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (255,255,255), thickness=0)
cv2.imshow('image_contour',img_contours)
cv2.waitKey(0)
# samples =  np.empty((0,10))
# responses = []
# keys = [i for i in range(48,58)]
#boxes=[]
# loop over the digit area candidates
for (i, c) in enumerate(contours_filtered):
    x,y,w,h = cv2.boundingRect(c)
    # box=x,y,w,h
    #boxes.append(box)
    img_contours_box=cv2.rectangle(resized, (x-10, y-10), (x + w+10, y + h+10), (36,255,12), 2)
    #cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0,255,0), 2)
    box=img_contours_box.copy()
    roi=box[y:y+h,x:x+w]
    
    print(roi.shape)
    cv2.imshow('roi',roi)
    cv2.waitKey(0)
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text=pytesseract.image_to_boxes(roi, config=custom_config)
    print(text)
    #roi_padded=np.pad(roi, ((5,5),(5,5)), "constant", constant_values=0)
    cv2.imwrite("image%04i.png" %i, roi)
    cv2.waitKey(0)
    
cv2.imshow('img_contours_box',img_contours_box )
cv2.waitKey(0)

# d = pytesseract.image_to_data(img_contours_box, output_type=Output.DICT)
# print(d.keys())
# keys = list(d.keys())
# run tesseract, returning the bounding boxes
#img_demo=cv2.imread('image0007.png')
# =============================================================================
# gray_demo=cv2.cvtColor(img_demo, cv2.COLOR_BGR2GRAY)
#  # Otsu Tresholding automatically find best threshold value
# _, binary_image = cv2.threshold(gray_demo, 0, 255, cv2.THRESH_OTSU)
# # invert the image if the text is white and background is black
# count_white = np.sum(binary_image > 0)
# count_black = np.sum(binary_image == 0)
# if count_black > count_white:
#     binary_image = 255 - binary_image
# =============================================================================
# =============================================================================
# custom_config = r'--oem 3 --psm 6 outputbase digits'
# text=pytesseract.image_to_boxes(img_demo, config=custom_config)# also include any config options you use
# =============================================================================

# test_img=cv2.imread("image0003.png")
# test_img_org=test_img.copy()
# test_img =cv2.resize(test_img, (64, 64)) 
# test_img =cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('testimage',test_img)
# cv2.waitKey(0)
# x=np.array(test_img[0])
# test_img = x.reshape(1,-1).astype(np.float64)
# prediction = classifier.predict(test_img)[0]
# print("I think that digit is: {}".format(prediction))
# cv2.imshow("Image", test_img_org)
# cv2.waitKey(0)



# =============================================================================
# def get_array(file):
#     img = cv2.imread(file)
#     basename = os.path.basename(file)
#     height, width, channels = img.shape
#     h = int(height/2)
#     w = int(width/2)
#     px  = img[h,w]
#     return np.array([basename, height, width, channels, px[0], px[1], px[2]])
# 
# a = get_array('image0001.png')
# b = get_array('image0002.png') 
# c = get_array('image0003.png') 
# d = get_array('image0004.png') 
# e = get_array('image0005.png') 
# f = get_array('image0006.png') 
# g = get_array('image0007.png') 
# h = get_array('image0008.png') 
# np.savetxt("stats.csv", (a, b,c,d,e,f,g,h), delimiter=",", fmt='%s')
# =============================================================================


# =============================================================================
# =============================================================================
# image=cv2.imread("image0006.png")
# grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
# #ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
# contours, _ = cv2.findContours(grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# preprocessed_digits = []
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     
#     # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
#     cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
#     
#     # Cropping out the digit from the image corresponding to the current contours in the for loop
#     digit = grey[y:y+h, x:x+w]
#     cv2.imshow("roi",digit)
#     cv2.waitKey(0)
#     
#     # Resizing that digit to (18, 18)
#     resized_digit = cv2.resize(digit, (18,18))
#     
#     # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
#     padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
#     
#     # Adding the preprocessed digit to the list of preprocessed digits
#     preprocessed_digits.append(padded_digit)
# print("\n\n\n----------------Contoured Image--------------------")
# plt.imshow(image, cmap="gray")
# plt.show()
#     
# inp = np.array(preprocessed_digits)
# for digit in preprocessed_digits:
#     prediction = model.predict(digit.reshape(1, 28, 28, 1))  
#     
#     print ("\n\n---------------------------------------\n\n")
#     print ("=========PREDICTION============ \n\n")
#     plt.imshow(digit.reshape(28, 28), cmap="gray")
#     plt.show()
#     print("\n\nFinal Output: {}".format(np.argmax(prediction)))
# =============================================================================
    
# =============================================================================
#     print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction))
#     
#     hard_maxed_prediction = np.zeros(prediction.shape)
#     hard_maxed_prediction[0][np.argmax(prediction)] = 1
#     print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
#     print ("\n\n---------------------------------------\n\n")
# 
# =============================================================================



