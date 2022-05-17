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
        depth = []
        norm = []
        rgb = []
        if self.imageReceived:
            for i in range(200):
                # depth
                img = self.imageDepth
                img *= 1000
                img[np.isnan(img)] = 0
                img.astype(np.uint64)
                depth.append(img)

                # depth norm
                img_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX) / 255
                img_norm = img_norm.astype(np.float32)
                norm.append(img_norm)

                # RGB
                img_rgb = self.imageRGB
                rgb.append(img_rgb)
                cv.waitKey(25)

            print('Saving videos...')
            save(os.path.join('allVids', 'depth.npy'), depth)
            save(os.path.join('allVids', 'norm.npy'), norm)
            save(os.path.join('allVids', 'rgb.npy'), rgb)


if __name__ == '__main__':
    # init
    rospy.init_node('cap', anonymous=False)
    # get topics
    camera = Video()
    # capture video
    camera.cap_video()
