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
    return thresh


# find desired video
cap = cv.VideoCapture('../dynamicCamera/videos/specialCases/difficult1.avi')

suc, prev = cap.read()

# capture previous frame as optical flow needs 2 frames to work
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

# start count as frame counter
count = 0

# loops through all frames and runs the functions to calculate the segmentation output
while True:
    suc, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find optical flow in frame
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    x, y = cv.split(flow)

    # set threshold for vector under 15 and above 30 for both x and y
    x[abs(x) < 15] = 0
    y[abs(y) < 15] = 0

    x[abs(x) > 30] = 0
    y[abs(y) > 30] = 0

    xInt = np.uint8(abs(x))
    yInt = np.uint8(abs(y))

    flow = cv.merge([x, y])

    prevgray = gray

    # show results
    cv.imshow('flow', draw_flow(gray, flow))
    cv.imshow('flow HSV', draw_hsv(flow, img))

    count += 1
    print(count)

    # code for saving frames for testing
    '''
    if count == 1:
        cv.imwrite('../turtleBot/testing/applyMask0.png', draw_hsv(flow, img))
        # cv.imwrite('../turtleBot/testing/highlighted0.png', draw_hsv(flow, img)[0])
        cv.imwrite('../turtleBot/testing/RGB0.png', img)
    
    if count == 50:
        cv.imwrite('../turtleBot/testing/applyMask1.png', draw_hsv(flow, img)[1])
        cv.imwrite('../turtleBot/testing/highlighted1.png', draw_hsv(flow, img)[0])
        cv.imwrite('../turtleBot/testing/RGB1.png', img)

    if count == 150:
        cv.imwrite('../turtleBot/testing/applyMask2.png', draw_hsv(flow, img)[1])
        cv.imwrite('../turtleBot/testing/highlighted2.png', draw_hsv(flow, img)[0])
        cv.imwrite('../turtleBot/testing/RGB2.png', img)

    if count == 152:
        cv.imwrite('../turtleBot/testing/applyMask3.png', draw_hsv(flow, img)[1])
        cv.imwrite('../turtleBot/testing/highlighted3.png', draw_hsv(flow, img)[0])
        cv.imwrite('../turtleBot/testing/RGB3.png', img)

    if count == 155:
        cv.imwrite('../turtleBot/testing/applyMask4.png', draw_hsv(flow, img)[1])
        cv.imwrite('../turtleBot/testing/highlighted4.png', draw_hsv(flow, img)[0])
        cv.imwrite('../turtleBot/testing/RGB4.png', img)
    '''
    key = cv.waitKey(5)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
