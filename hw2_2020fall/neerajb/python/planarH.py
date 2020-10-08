import numpy as np
from numpy import linalg as LA
import cv2
import random
import math

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

	A = np.empty(shape=(0,9))

	for i in range(len(x1)):
		arr = np.empty(shape=(1,9))

		#                 -x         -y          1  0  0  0  xu                 yu                 u 
		arr1 = np.array(([-x2[i][0], -x2[i][1], -1, 0, 0, 0, x2[i][0]*x1[i][0], x2[i][1]*x1[i][0], x1[i][0]]))
		A = np.vstack((A, arr1))
		#                 0  0  0  -x         -y          1  xv                 yv                 v 
		arr2 = np.array(([0, 0, 0, -x2[i][0], -x2[i][1], -1, x2[i][0]*x1[i][1], x2[i][1]*x1[i][1], x1[i][1]]))
		A = np.vstack((A, arr2))
	
	#ATA and find eigenvector cooresponding to least eigen value
	A_ = np.dot(np.transpose(A), A)
	w,v = LA.eig(A_)
	min_eig_index = np.argmin(w)
	H2to1= v[:,min_eig_index]
	H2to1 = H2to1.reshape(3,3).astype(float) #SHOULD THIS BE A FLOAT?

	return H2to1

def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	avgx1 = np.mean(x1, axis=0)
	avgx2 = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	shift_x1 = x1 - avgx1
	shift_x2 = x2 - avgx2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	maxval_x1 = np.max(shift_x1)
	maxval_x2 = np.max(shift_x2)

	norm_x1 = shift_x1/maxval_x1
	norm_x2 = shift_x2/maxval_x2

	#Similarity transform 1
	T1  = np.array(([[1/maxval_x1, 0, -avgx1[0]/maxval_x1],
					[0, 1/maxval_x1, -avgx1[1]/maxval_x1], 
					[0, 0, 1]]))

	#Similarity transform 2
	T2  = np.array(([[1/maxval_x2, 0, -avgx2[0]/maxval_x2],
					[0, 1/maxval_x2, -avgx2[1]/maxval_x2], 
					[0, 0, 1]]))

	#Compute homography
	H = computeH(norm_x1, norm_x2)

	#Denormalization
	H2to1_ = np.dot(LA.inv(T1), H)
	H2to1 = np.dot(H2to1_, T2)

	# a = np.dot(H2to1, np.append(x2[1], 1))
	# print(x1[1], a[0]/a[2], a[1]/a[2])

	return H2to1

# print(computeH_norm(x1,x2))
# H = computeH_norm(x1,x2)
# print(H)
# a = np.dot(H, [2,4,1])
# print(a[0]/a[2], a[1]/a[2])


def computeH_ransac(locs1, locs2, opts):
	count = 0
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters
	# the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	# for i in range(max_iters):

	#np.random.seed(3)
	max_count = 0
	inliers = []

	for j in range(max_iters):
		count = 0
		inliers = []
		
		# generate 4 random points
		indices = np.random.choice(len(locs1), 4, replace=False)
		p1_1 = locs1[indices[0]]
		p1_2 = locs1[indices[1]]
		p1_3 = locs1[indices[2]]
		p1_4 = locs1[indices[3]]

		p2_1 = locs2[indices[0]]
		p2_2 = locs2[indices[1]]
		p2_3 = locs2[indices[2]]
		p2_4 = locs2[indices[3]]

		x1 = np.array(([p1_1, p1_2, p1_3, p1_4]))
		x2 = np.array(([p2_1, p2_2, p2_3, p2_4]))

		# compute H given these 4 points
		H = computeH_norm(x1,x2)

		for i in range(len(locs1)):
		
			# compute distance 
			a = np.dot(H, np.append(locs2[i], 1))
			b = locs1[i]
			dist = computeDistance([a[0]/a[2], a[1]/a[2]] ,b)

			# build inlier list
			if dist < inlier_tol:
				count += 1
				inliers.append(1)
			else:
				inliers.append(0)
			# if better solution found, save this H
			if count > max_count:
				max_count = count
				bestH2to1 = H
				bestInliers = inliers
		
		#print(bestInliers)
	return bestH2to1, bestInliers

def computeDistance(x1, x2):
	dist = np.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
	return dist 

def compositeH(H2to1, template, img):

	warped_template = cv2.warpPerspective(template, H2to1, dsize = (img.shape[1],img.shape[0]))
  
	#print(template.shape[:2])
	white_template = 255 * np.ones(template.shape[:2], dtype=np.uint8)
	
	warped_white_template = cv2.warpPerspective(white_template, H2to1, dsize = (img.shape[1],img.shape[0]))

	warped_black_template = cv2.bitwise_not(warped_white_template)

	img_masked = cv2.bitwise_and(img, img, mask=warped_black_template)

	#Create a composite image after warping the template image on top
	#of the image using the homography
	composite_img = cv2.bitwise_or(warped_template, img_masked)
	# cv2.imshow('img', final_img)
	# cv2.waitKey(0)  
	
	return composite_img


