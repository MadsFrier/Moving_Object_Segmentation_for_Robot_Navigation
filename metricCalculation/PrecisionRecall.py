# import library
import cv2 as cv

# initialize sum variables for later use
sumPrecision = 0
sumRecall = 0

# read ground truth and prediction frames, and convert to grayscale
gt = cv.imread('testFrame/gtTest.png', 0)
pred = cv.imread('testFrame/predTest.png', 0)

(h, w) = gt.shape[:2]  # convert image to 2D matrix

# initialize all true and false positives and negative
TP = 0
TN = 0
FP = 0
FN = 0

# loop through all pixels
for x in range(h):
    for y in range(w):
        (gray) = gt[x, y]
        (gray1) = pred[x, y]

        if gray == gray1:
            if gray == 0:
                TN = TN + 1  # true negative
            else:
                TP = TP + 1  # true positive

        if gray != gray1:
            if gray == 0:
                FP = FP + 1  # false positive
            else:
                FN = FN + 1  # false negative

# print all values
print('True positive: ', TP)
print('True negative: ', TN)
print('False positive: ', FP)
print('False negative: ', FN)

# calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)

# print precision and recall
print('Precision: ', precision)
print('Recall: ', recall)
