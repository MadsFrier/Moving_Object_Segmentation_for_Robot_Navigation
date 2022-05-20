# import libraries
import numpy as np
import cv2 as cv


# draws vector field for optical flow
def draw_flow(img, flow, step=12):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(img_bgr, lines, 0, (0, 0, 255))

    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


# outputs optical flow as binary image
def draw_hsv(flow, img):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    grayNew = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    grayNew = cv.GaussianBlur(grayNew, (5, 5), 0)
    dst, thresh = cv.threshold(grayNew, 5, 255, cv.THRESH_BINARY)
    thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=8)

    backToRBG = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)  # convert gray to RBG

    applyMask = cv.bitwise_and(img, backToRBG)

    return applyMask


# find desired video
cap = cv.VideoCapture('../dynamicCamera/videos/results/crabHumanNoBg.avi')

suc, prev = cap.read()

prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

# loops through all frames and runs the functions to calculate the segmentation output
while True:
    suc, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    x, y = cv.split(flow)

    # determines camera direction (manually atm)
    cameraDirection = 'right'

    if cameraDirection == 'right':

        # Find the average for all vectors to the left (as this is right), and delete those vectors
        # within a certain range
        backGroundX = np.abs(x)
        backNegative = x[x < 0]
        negativeMeanX = np.mean(backNegative)
        negativeDeviationX = np.std(backNegative)
        medianNegativeX = np.median(backNegative)
        x[(x < 0) & (np.abs(x) < np.abs(negativeMeanX + negativeDeviationX * 5))] = 0
        x[(x < 0) & (np.abs(x) > np.abs(negativeMeanX + negativeDeviationX * 5))] = 0
        print('Min', np.abs(negativeMeanX) + negativeDeviationX)
        print('Max', np.abs(negativeMeanX + negativeDeviationX * 3))
    elif cameraDirection == 'left':

        # opposite for left
        backGroundX = np.abs(x)
        backPositive = x[x > 0]
        positiveMeanX = np.mean(backPositive)
        positiveDeviationX = np.std(backPositive)
        x[(x > 0) & (np.abs(x) > np.abs(positiveMeanX + positiveDeviationX * 3)) & (
                np.abs(x) < np.abs(positiveMeanX) + positiveDeviationX)] = 0
    elif cameraDirection == 'forward':
        # only looks at average, as this has opposite direction in 2D space
        averageX = np.mean(np.abs(x))
        deviationX = np.std(np.abs(x))
        x[(np.abs(x) > np.abs(averageX + deviationX * 3)) | (np.abs(x) < np.abs(averageX) + deviationX)] = 0
        averageY = np.mean(np.abs(y))
        deviationY = np.std(np.abs(y))
        y[(np.abs(y) > np.abs(averageY + deviationY * 3)) | (np.abs(y) < np.abs(averageY) + deviationY)] = 0

    averageY = np.mean(y)

    xInt = np.uint8(abs(x))
    yInt = np.uint8(abs(y))

    h, w = flow.shape[:2]

    flow = cv.merge([x, y])

    prevgray = gray
    
    # show results
    cv.imshow('flow', draw_flow(gray, flow))
    cv.imshow('flow HSV', draw_hsv(flow, img))

    key = cv.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
