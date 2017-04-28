'''
Coded by luke on 28th April 2017
Aiming to test the lane detection program for the self driving car

'''

# import the needed packages
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2

# reading the input image and show it in the fixed window
image = cv2.imread('images/cat.png')

#print the information about the image
print ('This image is :',type(image),'with dimensions:',image.shape)
plt.imshow(image)

import math

def grayscale(img):
	return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def canny(img,low_threshold,high_threshold):
	# image should be the gray scale image 
	return cv2.Canny(img,low_threshold,high_threshold)

def gaussian_blur(img,kernel_size):
	return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

def region_of_interest(img,vertices):
	"""
	Applies an image mask, only keeps the region of the image defiend by the polygon fromed from 'vertices'.The rest of the image is set to black

	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)
	
	# define a 3 channels or 1 channel color to full the mask with depending on the input  image 
	if len(img.shape) > 2:
		channel_count = img.shape[2] 
		ignore_mask_color = (255,) * channel_count

	else:
		ignore_mask_color = 255
	
	#filling pixels inside the polygon defined by 'vertices' with the fill color
	cv2.fillPoly(mask,vertices,ignore_mask_color)

	#return the image only where mask pixels are non zero
	masked_image = cv2.bitwise_and(img,mask)

	return masked_image
	
def draw_lines(img,lines,color=[255,0,0],thickness = 8):
		"""
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
	#define the use of global variables
	global last_right_gradient
	global last_left_gradient
	global last_bottom_right
	global last_top_right
	global last_bottom_left
	global last_top_left

	#define variables used for tweaking the lines
	alp = 0.2

	#Additional variables used for minor tweaks.Kept at 0 in current file
	#aas the tweaks were not necessary
	left_gradient_offset = 0
	right_gradient_offset = 0
	right_line_offset = 0
	left_line_offset = 0	
	
	#local variables used for keeping track of both left and right line which include
	#1.number of counts of left and right lines
	#2.total gradient for left and right lines
	#3.longest line in left and right lines
	left_count = 0
	left_gradient_total =0 
	left_longest = 0
	right_count = 0
	right_gradient_total = 0
	right_longest = 0

	#Run through all lines identified by hough transform
	if (lines is not True):
		for line in lines:
			for x1,y1,x2,y2 in line:
				#if x2 != x1 (not vertical lines ) gradient >0.4
				# consider the line as a right line and add to right_gradient_total	
				if (x2 ! x1 and float(y2-y1)/(x2-x1) > 0.4):
					#uncomment below to see all lines drawn
					#cv2.line(img,(x1,y1),(x2,y2),color,thickness)
					right_gradient_total += float(y2-y1)/(x2-x1)
					right_count += 1
							
					# find the lognest line on the right so that coordinates can be used for line extrapolation
					if (math.sqrt((y2-y1)**2 + (x2-x1)**2) > right_longest):
							right_longest = math.sqrt((y2-y1)**2+(x2-x1)**2)
							right_longest_line = line

				# if x2!
				else 
		
	
