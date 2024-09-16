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
import matplotlib.pyplot as plt

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

#-------------------------Reference Image Processing---------------------
# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread('digitsRef.png')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.adaptiveThreshold(ref,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
cv2.imshow('reference',ref)
cv2.waitKey(0)

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
 	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
sorted_contours = contours.sort_contours(refCnts, method="left-to-right")[0]
print("Number of Contours found in reference image = " + str(len(refCnts))) 
digits = {}
cv2.drawContours(ref, refCnts, -1, (0, 255, 0), 3)  
cv2.imshow('Contours', ref) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

# loop over the OCR-A reference contours
for (i, c) in enumerate(sorted_contours):
 	# compute the bounding box for the digit, extract it, and resize
 	# it to a fixed size
 	(x, y, w, h) = cv2.boundingRect(c)
 	roi = ref[y:y + h, x:x + w]
 	roi = cv2.resize(roi, (57, 88))
 	# update the digits dictionary, mapping the digit name to the ROI
 	digits[i] = roi
     
# for x in digits:
#  cv2.imshow('digits_roi',digits[x])
#  cv2.waitKey(0)
#-------------------------Ends Here--------------------------------------


# # initialize a rectangular (wider than it is tall) and square
# # structuring kernel
# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#-------------------------Actual Image processing------------------------
# load the input image, resize it, and convert it to grayscale
image = cv2.imread('image1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray, 0,255)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)
#-------------------------Ends Here--------------------------------------


#------------------Find Contour of Digits to crop------------------------
contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
print("Number of Contours found in actual image= " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
# cv2.imshow('Contours', image) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
 
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
x,y,w,h=cv2.boundingRect(sorted_contours[2])
cropped_contour=gray[y:y+h, x:x+w]
cv2.imshow('Image',cropped_contour)
cv2.waitKey(0)
#-------------------------Ends Here---------------------------------------


#-----------------image processing on cropped Image-----------------------

(origH, origW) = cropped_contour.shape[:2]
# set the new width and height and then determine the ratio in change
# for both the width and height
scale_percent = 200 # percent of original size
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

# thresholding the image
threshold_img = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('threshold image', threshold_img) 
cv2.waitKey(0)
cv2.destroyAllWindows()

#Dilating the image
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# Appplying dilation on the threshold image
dilation = cv2.dilate(threshold_img, rect_kernel, iterations = 1)
cv2.imshow('dilate image', dilation) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# Canny edge detection

crop_image_edges = cv2.Canny(dilation,150,150, apertureSize=3, L2gradient=True)
cv2.imshow('Edged_cropped',crop_image_edges)
cv2.waitKey(0)
# # edges = cv2.Canny(img, 235, 250)
plt.imshow(crop_image_edges, cmap='gray')
# plt.imshow(crop_image_edges1, cmap='gray')

contours, hierarchy = cv2.findContours(crop_image_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_dict = dict()
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    area = cv2.contourArea(cont)
    if 5< area and 5 < w and h > 3:
        contours_dict[(x, y, w, h)] = cont

print("Number of Contours found in cropped Image= " + str(len(contours))) 
contours_filtered = sorted(contours_dict.values(), key=cv2.boundingRect)
print("Number of filtered Contours found in cropped Image= " + str(len(contours_filtered))) 

#blank_background = np.zeros_like(crop_image_edges)
img_contours = cv2.drawContours(resized, contours_filtered, -1, (255,255,255), thickness=2)
# # img_contours = cv2.drawContours(blank_background, contours_filtered, -1, (100,100,100), thickness=cv2.FILLED)
#plt.axis('off')
# cv2.imshow('image_contour',img_contours)
# cv2.waitKey(0)
boxes=[]
# loop over the digit area candidates
for (i, c) in enumerate(contours_filtered):
    x,y,w,h = cv2.boundingRect(c)
    box=x,y,w,h
    #print("Area of a contour is= "+ str(cv2.contourArea(c))) 
    boxes.append(box)
    img_contours_box=cv2.rectangle(resized, (x, y), (x + w, y + h), (36,255,12), 2)
    #cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0,255,0), 2)
cv2.imshow('img_contours_box',img_contours_box )
cv2.waitKey(0)
# sort the digit locations from left-to-right, then initialize the
# list of classified digits
boxes = sorted(boxes, key=lambda x:x[0])

groupOutput = []
# loop over the digit contours
for c in boxes:
		# compute the bounding box of the individual digit, extract
		# the digit, and resize it to have the same fixed size as
		# the reference OCR-A images
		(x, y, w, h) = cv2.boundingRect(c)
		roi = img_contours_box[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		# initialize a list of template matching scores	
		scores = []
		# loop over the reference digit name and digit ROI
		for (digit, digitROI) in digits.items():
			# apply correlation-based template matching, take the
			# score, and update the scores list
			result = cv2.matchTemplate(roi, digitROI,
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)
		# the classification for the digit ROI will be the reference
		# digit name with the *largest* template matching score
		groupOutput.append(str(np.argmax(scores)))
        
    




# # # initialize the list of results
# # results = []

# # x = int(x * rW)
# # y = int(y * rH)
# # w = int(w * rW)
# # h = int(h * rH)
# # padding=0.0
# # # in order to obtain a better OCR of the text we can potentially
# # # apply a bit of padding surrounding the bounding box -- here we
# # # are computing the deltas in both the x and y directions
# # dX = int((w - x) * padding)
# # dY = int((y - h) * padding)
# # # apply padding to each side of the bounding box, respectively
# # x = max(0, x - dX)
# # y = max(0, y - dY)
# # w = min(origW, w+ (dX * 2))
# # h = min(origH, h + (dY * 2))
# # # extract the actual padded ROI
# # roi = img_contours[y:h, x:w]
# # # in order to apply Tesseract v4 to OCR text we must supply
# # # (1) a language, (2) an OEM flag of 4, indicating that the we
# # # wish to use the LSTM neural net model for OCR, and finally
# # # (3) an OEM value, in this case, 7 which implies that we are
# # # treating the ROI as a single line of text
# # config = ("-l eng --oem 1 --psm 7")
# # text = pytesseract.image_to_string(roi, config=config)
# # # add the bounding box coordinates and OCR'd text to the list
# # # of results
# # results.append(((x, y, w, h), text))





# # roi_resize ndY= cv2.resize(roi,(28, 28), interpolation = cv2.INTER_CUBIC)
# # plt.imshow(roi_resize, cmap='gray')



# # config = ("-l eng --oem 1 --psm 7")
# # text = pytesseract.image_to_string(img_contours, config=config)
# # #text = "".join([c if ord(c) < 128 else "" for c in text]).strip()


# # def is_overlapping_horizontally(box1, box2):
# #     x1, _, w1, _ = box1
# #     x2, _, _, _ = box2
# #     if x1 > x2:
# #         return is_overlapping_horizontally(box2, box1)
# #     return (x2 - x1) < w1

# # def merge(box1, box2):
# #     assert is_overlapping_horizontally(box1, box2)
# #     x1, y1, w1, h1 = box1
# #     x2, y2, w2, h2 = box2
# #     x = min(x1, x2)
# #     w = max(x1 + w1, x2 + w2) - x
# #     y = min(y1, y2)
# #     h = max(y1 + h1, y2 + h2) - y
# #     return (x, y, w, h)

# # def windows(contours):
# #     """return List[Tuple[x: Int, y: Int, w: Int, h: Int]]"""
# #     boxes = []
# #     for cont in contours:
# #         box = cv2.boundingRect(cont)
# #         if not boxes:
# #             boxes.append(box)
# #         else:
# #             if is_overlapping_horizontally(boxes[-1], box):
# #                 last_box = boxes.pop()
# #                 merged_box = merge(box, last_box)
# #                 boxes.append(merged_box)
# #             else:
# #                 boxes.append(box)
# #     return boxes

# # boxes = windows(contours_filtered)

# # img_box = resized.copy()
# # for box in boxes:
# #     x, y, w, h = box
# #     img_box = cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
# # plt.imshow(img_box)

# # x, y, w, h = boxes[0]
# # plt.axis('off')
# # roi = img_contours[y:y+h, x:x+w]
# # plt.imshow(roi)
# # custom_config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
# # result=pytesseract.image_to_string(roi, config=custom_config)