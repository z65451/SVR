import cv2

import pyCAIR

import os
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.filters import convolve
from tqdm import trange

from pyCAIR.helpers import writeImage as wI
from pyCAIR.helpers import createFolder as cF
from pyCAIR.helpers import writeImage as wI
from pyCAIR.helpers import writeImageG as wIG

def getEnergy(image):
    	
	filter_x = np.array([
		[1.0, 2.0, 1.0],
		[0.0, 0.0, 0.0],
		[-1.0, -2.0, -1.0],
		])

	filter_x = np.stack([filter_x] * 3, axis = 2)

	filter_y = np.array([
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0],
		])

	filter_y = np.stack([filter_y] * 3, axis = 2)

	image = image.astype('float32')

	convoluted = np.absolute(convolve(image, filter_x)) + np.absolute(convolve(image, filter_y))

	energy_map = convoluted.sum(axis = 2)

	return energy_map

def getMaps(image):
	rows, columns, _ = image.shape
	energy_map = getEnergy(image)
	energy_map = cv2.imread("attn.jpg")
	energy_map = cv2.cvtColor(energy_map, cv2.COLOR_BGR2GRAY)


	current_map = energy_map.copy()
	goback = np.zeros_like(current_map, dtype = np.int)

	for i in range(1, rows):
		for j in range(0, columns):
			if j == 0:
				min_index = np.argmin(current_map[i - 1, j : j + 2])
				goback[i, j] = min_index + j
				min_energy = current_map[i - 1, min_index + j]

			else:
				min_index = np.argmin(current_map[i - 1, j - 1 : j + 2])
				goback[i, j] = min_index + j -1
				min_energy = current_map[i - 1, min_index + j - 1]

			current_map[i, j] += min_energy

	return current_map, goback

def drawSeam(image):

	rows, columns, _ = image.shape
	cMap, goback = getMaps(image)

	mask = np.ones((rows, columns), dtype = np.bool)

	j = np.argmin(cMap[-1])

	for i in reversed(range(rows)):
		mask[i, j] = False
		j = goback[i, j]

	mask = np.logical_not(mask)
	image[...,0][mask] = 0 
	image[...,1][mask] = 0
	image[...,2][mask] = 255

	return image

def carve(image):

	rows, columns, _ = image.shape
	cMap, goback = getMaps(image)

	mask = np.ones((rows, columns), dtype = np.bool)

	j = np.argmin(cMap[-1])

	for i in reversed(range(rows)):
		mask[i, j] = False
		j = goback[i, j]

	mask = np.stack([mask] * 3, axis = 2)
	image = image[mask].reshape((rows, columns - 1, 3))
	
	return image


# from pyCAIR import generateEnergyMap, cropByColumn
from seamc2 import  cropByColumn
im = cv2.imread("img.jpg")

# from pyCAIR import cropByColumn, carve
aa, bb=cropByColumn(im, display_seams=1)

cv2.imwrite("carved.jpg", aa)
cv2.imwrite("carved2.jpg", bb)

a=1