#!/usr/bin/env python3

import rospy
import numpy as np
from os import uname
from re import match
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge


class RedDetectionNode(DTROS):
    def __init__(self, node_name):
        super(RedDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Image processing topics
        self._undistort_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"
        self._colour_detection_topic = f"/{self._vehicle_name}/red_detection_node/image/compressed"
        self._homography_param = f"/{self._vehicle_name}/updated_extrinsics"

        # Publishers and subscribers
        self.sub = rospy.Subscriber(
            self._undistort_topic, CompressedImage, self.camera_reader_callback, queue_size=1)

        # Publisher for visualization
        self.img_pub = rospy.Publisher(
            self._colour_detection_topic, CompressedImage, queue_size=1)

        # Publisher for red line distance
        self.distance_pub = rospy.Publisher(
            f"/{self._vehicle_name}/red_line_distance", Float32, queue_size=1)

        self._bridge = CvBridge()
        self._homography = np.array(rospy.get_param(
            self._homography_param, default=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])).reshape(3, 3)

        # Kernel for morphological operations
        self.kernel = np.ones((3, 3), "uint8")

        # Red color range in HSV
        # OpenCV HSV: H is 0-179, S is 0-255, V is 0-255
        self.red_lower = (0, 50, 50)
        self.red_upper = (20, 255, 255)

        rospy.loginfo("Red detection node initialized")

    def detect_red(self, hsvFrame):
        """Detect red color in HSV image"""
        mask = cv2.inRange(hsvFrame, np.array(
            self.red_lower, dtype=np.uint8), np.array(self.red_upper, dtype=np.uint8))
        mask = cv2.dilate(mask, self.kernel)
        return mask

    def get_distance_to_point(self, u, v):
        """Calculate the distance to point in world coordinates using homography"""
        pixel_coords = np.array([u, v, 1]).reshape(3, 1)
        world_coords = np.dot(self._homography, pixel_coords)

        # Convert from homogeneous coordinates to (X, Y)
        X = world_coords[0] / world_coords[2]
        Y = world_coords[1] / world_coords[2]
        euclidean_dist = (X**2 + Y**2)**(1/2)

        return X, Y, euclidean_dist

    def camera_reader_callback(self, msg):
        if rospy.is_shutdown():
            return

        # Convert the compressed image to an OpenCV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Convert the image to HSV
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a copy for visualization
        output_image = image.copy()

        # Detect red
        mask = self.detect_red(hsvFrame)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize distance to a large value
        distance = float('inf')

        # If contours are found, find the largest one
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only consider large contours
            # Reduced threshold for smaller red lines
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(output_image, (x, y),
                              (x + w, y + h), (0, 0, 255), 2)

                # Compute center of the bounding box
                center_x, center_y = x + w // 2, y + h // 2

                # Draw center point as a small circle
                cv2.circle(output_image, (center_x, center_y),
                           5, (255, 0, 0), -1)  # Blue dot

                # Calculate distance
                _, _, distance = self.get_distance_to_point(center_x, center_y)

                # Convert distance to a float value if it's a numpy array
                if isinstance(distance, np.ndarray):
                    distance = float(distance)

                # Add text showing the distance
                cv2.putText(output_image, f"Dist: {distance:.3f}m",
                            (center_x - 60, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Publish the distance
        # Ensure distance is a float before publishing
        if isinstance(distance, np.ndarray):
            distance = float(distance)
        self.distance_pub.publish(Float32(distance))

        # Publish the visualization image
        image_message = self._bridge.cv2_to_compressed_imgmsg(output_image)
        self.img_pub.publish(image_message)


if __name__ == '__main__':
    node = RedDetectionNode(node_name='red_detection_node')
    rospy.spin()
