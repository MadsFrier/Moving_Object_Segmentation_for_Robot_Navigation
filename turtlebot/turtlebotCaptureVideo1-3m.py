# import libraries
from __future__ import print_function  #
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
from numpy import save


# subscribe to RGB and depth topics on the turtlebot
class Video:
    def __init__(self):
        self.bridge = CvBridge()
        self.imageReceived = False

        imgTopicDepth = "/camera/depth_registered/hw_registered/image_rect_raw"
        imgTopicRGB = "/camera/rgb/image_rect_color"

        self.image_sub = rospy.Subscriber(imgTopicDepth, Image, self.callback_depth)
        self.image_sub = rospy.Subscriber(imgTopicRGB, Image, self.callback_rgb)

        rospy.sleep(1)

    # Convert topic data to openCV format (array) for depth
    def callback_depth(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'passthrough')
        except CvBridgeError as e:
            print(e)

        self.imageReceived = True

        self.imageDepth = cv_image

    # Convert topic data to openCV format (array) for RGB
    def callback_rgb(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        self.imageReceived = True
        self.imageRGB = cv_image

    # capture depth and RGB frames for 200 loops and save the .npy file for depth, depth norm, and RGB
    def cap_video(self):
        if self.imageReceived:

            # prepare to save videos
            frameSize = (640, 480)
            outDepth = cv.VideoWriter('difficult1.avi', cv.cv.CV_FOURCC(*'XVID'), 12, frameSize)
            outRGB = cv.VideoWriter('output_video1.avi', cv.cv.CV_FOURCC(*'XVID'), 12, frameSize)
            
            #loop through 200 frames and save depth and RGB from 1-3 m
            for x in range(200):
                img = self.imageDepth
                img = img.copy()
                imgRGB = self.imageRGB
                img[np.isnan(img)] = 0
                img[img < 1] = 0
                img[img > 3] = 0
                img[img != 0] = 255
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

                img = img.astype(np.uint8)
                imgRGB = imgRGB.astype(np.uint8)

                depth1to3 = cv.bitwise_and(img, imgRGB)
                outDepth.write(depth1to3)
                outRGB.write(imgRGB)

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

            outDepth.release()
            outRGB.release()
            print('Video has been saved!')


if __name__ == '__main__':
    # init
    rospy.init_node('cap', anonymous=False)
    # get topics
    camera = Video()
    # capture video
    camera.cap_video()

