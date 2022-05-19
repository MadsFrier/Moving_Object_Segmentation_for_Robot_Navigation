import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from numpy import load
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist
import collections

current_id = 255

# find midpoint of object
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


allPath = 'C:/Users/madsf/PycharmProjects/Moving Object Segmentation/softwareDesign/turtleBot/allVids/'
# C:/Users/madsf/PycharmProjects/Moving Object Segmentation/softwareDesign/turtleBot/allVids

depthVid = load(allPath + 'depth.npy')
normVid = load(allPath + 'norm.npy')
rgbVid = load(allPath + 'rgb.npy')


def draw_hsv(flow, img, depth):
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
    dst, thresh = cv.threshold(grayNew, 5, 255, cv.THRESH_BINARY)

    edge = cv.Canny(thresh, 50, 100)

    # dilate twice and erode to close gaps in edges (closing)
    edge = cv.dilate(edge, None, iterations=2)
    edge = cv.erode(edge, None, iterations=1)

    # contours
    cnts = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts != []:

        c = max(cnts, key=cv.contourArea)

        orig = thresh.copy()
        box = cv.minAreaRect(c)
        box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        [centerX, centerY] = midpoint((tltrX, tltrY), (blbrX, blbrY))

        centerX = int(centerX)
        centerY = int(centerY)

        TARGET = (centerX, centerY)

        def find_nearest_white(img, target):
            nonzero = cv.findNonZero(img)
            distances = np.sqrt((nonzero[:, :, 0] - target[0]) ** 2 + (nonzero[:, :, 1] - target[1]) ** 2)
            nearest_index = np.argmin(distances)
            return nonzero[nearest_index]

        seedPoint = find_nearest_white(thresh, TARGET)
        list(seedPoint)

        print(depth[seedPoint[0][0]][seedPoint[0][1]])

        burn_queue = collections.deque()

        if depth[seedPoint[0][0]][seedPoint[0][1]] != 0:
            def ignite_fire(mask, pixel_x, pixel_y, depthInterval):
                burn_queue.append((pixel_x, pixel_y))
                while len(burn_queue) > 0:
                    global current_id
                    current_pos = burn_queue.pop()
                    mask[current_pos] = current_id
                    north = current_pos[0], current_pos[1] - 1
                    west = current_pos[0] - 1, current_pos[1]
                    south = current_pos[0], current_pos[1] + 1
                    east = current_pos[0] + 1, current_pos[1]

                    if depthInterval[0] < mask[north] < depthInterval[1]:
                        burn_queue.append(north)

                    if depthInterval[0] < mask[west] < depthInterval[1]:
                        burn_queue.append(west)

                    if depthInterval[0] < mask[south] < depthInterval[1]:
                        burn_queue.append(south)

                    if depthInterval[0] < mask[east] < depthInterval[1]:
                        burn_queue.append(east)

                return depth

            depthInter = [depth[seedPoint[0][0]][seedPoint[0][1]] - 100, depth[seedPoint[0][0]][seedPoint[0][1]] + 100]

            ignite_fire(depth, seedPoint[0][0], seedPoint[0][1], depthInter)

            depth[depth != 255] = 0

            cv.imshow('depth', depth)
            cv.imwrite('clusterMask1.png', depth)
            cv.imwrite('clusterRGB1.png', img)
            cv.waitKey(0)

    return thresh


prev = rgbVid[0, :, :]

prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

h, w = normVid.shape[:2]

for i in range(41, h):  # 41 chair 79 person
    img = rgbVid[i, :, :]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    copy = flow.copy()
    newx, newy = cv.split(copy)
    x, y = cv.split(flow)
    vC = np.sqrt(x * x + y * y)
    h, w = vC.shape
    vC = np.ravel(vC)
    vC = vC.reshape(-1, 1)

    depthC = depthVid[i, :, :]
    depthC = np.ravel(depthC)
    depthC = depthC.reshape(-1, 1)

    depthCLuster = depthVid[i, :, :]
    depthCLuster = np.ravel(depthCLuster)
    depthCLuster = depthCLuster.reshape(-1, 1)
    depthC = depthC * 0.010
    km = KMeans(n_clusters=2)
    y_predicted = km.fit_predict(depthC, vC)

    y_predicted = y_predicted.reshape(-1, 1)
    features = np.hstack((depthC, vC, y_predicted))
    type1 = features[features[:, 2] == 0]
    type2 = features[features[:, 2] == 1]

    xGt = x
    yGt = y

    # add if statement to take highest average instead of always the red one
    newflow = features
    if np.mean(type1[:, 1]) >= np.mean(type2[:, 1]):
        testF = newflow
        newflow[newflow[:, 2] != 0] = 0

        testF[testF[:, 2] != 0] = 127

        testF = testF[:, 0]
        c127 = testF[testF == 127].shape[0]
        testF[testF == 127] = 0

        nn, hh, ww = normVid.shape[:3]
        testF = testF.reshape(480, 640)  # cv.imshow('Clust Test', testF)  # cv.waitKey(400)
    else:
        testF = newflow
        newflow[newflow[:, 2] != 1] = 0

        testF[testF[:, 2] != 1] = 127
        testF[testF != 127] = 255
        testF = testF[:, 0]
        c127 = testF[testF == 127].shape[0]
        testF[testF == 127] = 0

        nn, hh, ww = normVid.shape[:3]

        testF = testF.reshape(480, 640)

        # cv.imshow('Clust Test', testF)  # v.waitKey(400)

    depthC = newflow[:, 0]
    depthC = depthC.reshape(h, w)
    # test flow
    xcopy = x
    xcopy = np.where(depthC == 0, 0, xcopy)
    ycopy = y
    ycopy = np.where(depthC == 0, 0, ycopy)

    flow = cv.merge([x, y])
    clustflow = flow
    clustflow = cv.merge(([xcopy, y]))
    prevgray = gray

    # cv.imshow('flow HSV', draw_hsv(flow, img, depthVid[i, :, :]))
    cv.imshow('flow Clustering', draw_hsv(clustflow, img, depthVid[i, :, :]))
    # cv.imshow('norm', normVid[i, :, :])
    cv.imshow('rgb', rgbVid[i, :, :])
    cv.waitKey(30)

    print(i)
    key = cv.waitKey(5)
    if key == ord('q'):
        break

cv.destroyAllWindows()
