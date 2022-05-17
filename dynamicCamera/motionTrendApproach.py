import math
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt



#linear: 0.2m/s
#angular: 0.5rad/s


def draw_flow(img, flow, step=12):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 0, 255))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow, img):

    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    grayNew = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    #grayNew = cv2.GaussianBlur(grayNew, (9,9), 0)
    grayNew = cv2.GaussianBlur(grayNew, (5, 5), 0)
    dst, thresh = cv2.threshold(grayNew, 5, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=8)

    backToRBG = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)  # convert gray to RBG


    applyMask = cv2.bitwise_and(img, backToRBG)

    return applyMask


cap = cv2.VideoCapture('../Turtlebot/crabHumanNoBg.avi')

suc, prev = cap.read()

prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:

    suc, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    x, y = cv2.split(flow)
    print(x.shape)

    cameraDirection = 'right'

    if cameraDirection == 'right':

        backGroundX = np.abs(x)
        backNegative = x[x < 0]
        negativeMeanX = np.mean(backNegative)
        negativeDeviationX = np.std(backNegative)
        medianNegativeX= np.median(backNegative)
        x[(x<0)& (np.abs(x)<np.abs(negativeMeanX+ negativeDeviationX*5)) ]=0
        x[(x < 0) & (np.abs(x) > np.abs(negativeMeanX + negativeDeviationX * 5))] = 0
        #y[(y<0)|(y>0)]=0
        print('Min',np.abs(negativeMeanX)+ negativeDeviationX )
        print('Max', np.abs(negativeMeanX+ negativeDeviationX*3))
    elif cameraDirection == 'left':
        backGroundX = np.abs(x)
        backPositive = x[x > 0]
        positiveMeanX = np.mean(backPositive)
        positiveDeviationX = np.std(backPositive)
        x[(x > 0) & (np.abs(x) > np.abs(positiveMeanX + positiveDeviationX * 3)) & (np.abs(x) < np.abs(positiveMeanX) + positiveDeviationX)] = 0
    elif cameraDirection == 'forward':
        averageX = np.mean(np.abs(x))
        deviationX = np.std(np.abs(x))
        x[ (np.abs(x) > np.abs(averageX+ deviationX * 3)) | (np.abs(x) < np.abs(averageX) + deviationX )] = 0
        averageY = np.mean(np.abs(y))
        deviationY = np.std(np.abs(y))
        y[ (np.abs(y) > np.abs(averageY + deviationY * 3)) | (np.abs(y) < np.abs(averageY) + deviationY)] = 0


    averageY = np.mean(y)

    xInt = np.uint8(abs(x))
    yInt = np.uint8(abs(y))

    #print(xInt[300][300])

    h, w = flow.shape[:2]

    flow = cv2.merge([x, y])

    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    #print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))
    cv2.imshow('flow HSV', draw_hsv(flow, img))

    key = cv2.waitKey(5)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
