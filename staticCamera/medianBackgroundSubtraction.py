# import libraries
import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture('../softwareDesign/groundtruth/outside1.mp4')  # capture webcam footage
# "../softwareDesign/media/vtest.avi"  # video

updateVal = 100  # background update frequency

skipVal = 4  # frame skip interval

n = 0
x = 0

check = 0

frames = []  # make array to hold frames to calc median frame

while cap.isOpened():

    startTime1 = time.time()
    ret, frame = cap.read()  # read one frame
    frames.append(frame)  # add frame to array
    x = x + 1  # keep count of frame number

    if x == 1:  # set first background as first frame
        medianFrame = frame
        grayMedianFrame = cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY)

    stopTime1 = time.time()

    if x > updateVal + n:  # takes median of all frames in frames array
        frames = frames[::skipVal]  # the amount of frames to skip
        print(len(frames))
        medianTime = time.time()
        medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
        grayMedianFrame = cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY)  # grayscale median frame
        frames = []
        n = n + updateVal
        medianStopTime = time.time()
        print(f"The execution time is: {medianStopTime - medianTime}")

    startTime2 = time.time()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # grayscale frame

    sub = cv.absdiff(gray, grayMedianFrame)  # subtract background from frame

    (h, w) = medianFrame.shape[:2]  # convert image to 2D matrix
    (h1, w1) = gray.shape[:2]  # convert image to 2D matrix

    blur = cv.blur(sub, (3, 3), 0)  # blur frame

    dst, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY_INV)  # threshold frame

    median = cv.medianBlur(thresh, 3)  # apply noice filter (median blur)

    morph = cv.morphologyEx(median, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    inv = cv.bitwise_not(morph)

    backToRBG = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)  # convert gray to RBG

    black = np.where(backToRBG == 0)

    backToRBG[black[0], black[1], :] = [0, 0, 255]

    applyMask = cv.bitwise_and(frame, backToRBG)  # use and logic operation to apply the mask to the current frame
    ''''# --------------------------------------------------------------------- #
    # display frame count
    cv.rectangle(applyMask, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(applyMask, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # --------------------------------------------------------------------- #'''

    cv.imshow('w/ mask', applyMask)  # show mask frame
    cv.imshow('original', frame)  # show estimated background
    cv.imshow('inv', inv)  # show estimated background

    stopTime2 = time.time()
    # print(f"The execution time is: {(stopTime1 - startTime1) + (stopTime2 - startTime2)}")

    if x == 320:
        cv.imwrite('groundTruth/outside1Tests/predTest.png', inv)

        print('Done saving desired frame!')


    if cv.waitKey(25) & 0xFF == ord('q'):  # 25 msec until next frame
        break
