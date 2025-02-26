#!/usr/bin/env python3


import rospy
from os import uname
from re import match
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image

import cv2
from cv_bridge import CvBridge


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._hostname = uname()[1]

        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._custom_topic = f"/annotated_image"
        self.radius = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/radius')
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # construct subscriber
        self.sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.callback)
        self.pub = rospy.Publisher(self._custom_topic, Image, queue_size=10)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        width, height, _ = image.shape
        print("width:", width, "height:", height)
        print("radius", self.radius)

        # convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        size_text = f"{width}X{height}"

        # Format the annotation text
        annotation_text = f"Duck {self._hostname} says, 'Cheese! Capturing {size_text} - quack-tastic!'"

        # Annotate the image
        annotated_image = cv2.putText(
            gray_image,
            annotation_text,
            (10, 100),  # Bottom-left corner for the text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font size
            (0, 0, 0),  # White text for grayscale image
            1,  # Thickness
            cv2.LINE_AA
        )

        # Convert back to ROS Image message
        annotated_msg = self._bridge.cv2_to_imgmsg(annotated_image)

        self.pub.publish(annotated_msg)


if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
