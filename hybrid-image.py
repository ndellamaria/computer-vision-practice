import numpy as np
import cv2 as cv

def filter(image, filter):
	return cv.filter2D(src=image, ddepth=-1, kernel=filter)


def hybrid(img1,img2):
	pass


dog = cv.imread('dog.jpg')
cat = cv.imread('cat.jpg')
blur_filter = np.ones((3,3))
blur_filter /= np.sum(blur_filter)

blur_dog = filter(dog,blur_filter)

blur_cat = filter(cat,blur_filter)

cf = 50

blur = cv.GaussianBlur(cat,(cf*4+1,cf*4+1),sigmaX=cf,sigmaY=cf)

high_cat = cv.subtract(cat,blur)
# high_cat = cv.add(high_cat,0.5)

cf2 = 5
low_dog = cv.GaussianBlur(dog,(cf2*4+1,cf2*4+1),sigmaX=cf2,sigmaY=cf2)

sobel = np.array([
[-1,0,1],[-2,0,2],[-1,0,1]
])

sobel_cat = filter(cat,blur_filter)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',low_dog)
cv.waitKey(0)
cv.destroyAllWindows()

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',high_cat)
cv.waitKey(0)
cv.destroyAllWindows()

hybrid = cv.add(low_dog,high_cat)

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.imshow('image',hybrid)
cv.waitKey(0)
cv.destroyAllWindows()

