# import libraries
import cv2 as cv
import numpy as np

# read ground truth and prediction frames
gt = cv.imread('testFrame/gtTest.png')
pred = cv.imread('testFrame/predTest.png')

# calculate intersection
intersection = cv.bitwise_and(gt, pred)

# calculate union
union = cv.bitwise_or(gt, pred)

# subtract ground truth and prediction
sub = cv.subtract(gt, pred)

# display subtraction, intersection and union of the 2 frames
cv.imshow('subtract', sub)
cv.imshow('intersection', intersection)
cv.imshow('union', union)

# calculate IoU score
IoUScore = np.sum(intersection) / np.sum(union)

# print IoU score
print(IoUScore)

cv.waitKey(0)
