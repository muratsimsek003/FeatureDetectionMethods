'Method1: Haris Corner detection'
# Importing the libraries
import cv2
import numpy as np
image = cv2.imread('cameraman.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = np.float32(gray_image)
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)
image[dst > 0.01 * dst.max()] = [0, 255, 0]
cv2.imshow('haris_corner', image)
cv2.waitKey()

'Method 2: Shi-Tomasi corner detection'

import cv2
import numpy as np

# Reading the image and converting into B?W
image = cv2.imread("cameraman.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(
	gray_image, maxCorners=50, qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)
for item in corners:
	x, y = item[0]
	x = int(x)
	y = int(y)
	cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
# Showing the image
cv2.imshow('Shi_Tomasi', image)
cv2.waitKey()

'Method 3: SIFT (Scale-Invariant Feature Transform)'
# Importing the libraries
import cv2
#
# # Reading the image and converting into B/W
image = cv2.imread('cameraman.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying the function
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray_image, None)


# Applying the function
kp_image = cv2.drawKeypoints(image, kp, None, color=(
	0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', kp_image)
cv2.waitKey()

'Method 4: FAST algorithm for corner detection'

import cv2

# Reading the image and converting into B/W
image = cv2.imread('cameraman.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
kp = fast.detect(gray_image, None)
kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
cv2.imshow('FAST', kp_image)
cv2.waitKey()

'Method 5: ORB (Oriented FAST and Rotated Brief)'

import cv2


image = cv2.imread('cameraman.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_image, None)
kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
cv2.imshow('ORB', kp_image)
cv2.waitKey()
