#!/usr/bin/env python3

import rospy
import time
from os import uname
from re import match
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from odometry.srv import SetLEDColor
import cv2
from cv_bridge import CvBridge
from enum import Enum
from math import pi


class SectionID(Enum):
    STRAIGHT = 1
    CURVE = 2


class ColourDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ColourDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # image processing topics
        self._undistort_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"
        self._colour_detection_topic = f"/{self._vehicle_name}/colour_detection_node/image/compressed"
        self._homography_param = f"/{self._vehicle_name}/updated_extrinsics"

        # movement topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        # Encoder data and flags
        self._ticks_left = None
        self._ticks_right = None
        self._start_left = 0
        self._start_right = 0
        self._just_started_left = True
        self._just_started_right = True
        self._moving_forward = True
        self._rotating_cw = True
        self._velocity = 0.50
        self.distance = 1000.0
        self.detected_color = None

        # Subscribe to encoder topics
        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Get parameters
        self._radius = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)  # Default radius
        self._resolution_left = rospy.get_param(
            f'/{self._vehicle_name}/left_wheel_encoder_node/resolution', 135)
        self._resolution_right = rospy.get_param(
            f'/{self._vehicle_name}/right_wheel_encoder_node/resolution', 135)

        # this is just equal to 2dw, so no need to change it for now
        self._baseline = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/baseline', 0.1)

        if self._resolution_right != self._resolution_left:
            rospy.logwarn(
                "The resolutions of the left and right wheels do not match!")

        self._dist_per_tick = self._radius * 2 * pi / self._resolution_left

        # Initialize velocities
        self._vel_left = self._velocity
        self._vel_right = self._velocity
        self._rate = rospy.Rate(10)

        # Publisher for wheel commands
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheels_publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        self._bridge = CvBridge()

        # Wait for the service to be available before proceeding
        rospy.wait_for_service(f"/{self._vehicle_name}/set_led_color")
        self.set_led_color_service = rospy.ServiceProxy(
            f"/{self._vehicle_name}/set_led_color", SetLEDColor)

        self.sub = rospy.Subscriber(
            self._undistort_topic, CompressedImage, self.camera_reader_callback, queue_size=5)
        self.pub = rospy.Publisher(
            self._colour_detection_topic, CompressedImage, queue_size=5)
        self._homography = np.array(rospy.get_param(
            self._homography_param, default=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])).reshape(3, 3)

        self.kernel = np.ones((3, 3), "uint8")  # Optimized kernel size

        # Convert HSV values from 0-360 to OpenCV 0-179 range
        self.color_ranges = {
            "Red": ((0, 50, 50), (20, 255, 255), (0, 0, 255)),
            "Green": ((35, 60, 149), (85, 255, 255), (0, 255, 0)),
            "Blue": ((100, 100, 100), (140, 255, 255), (255, 0, 0)),
            "Yellow": ((25, 50, 50), (40, 255, 255), (0, 255, 255)),
            "White": ((0, 0, 200), (180, 50, 255), (255, 255, 255)),
        }

        # self.change_led_color(["green", "green", "green", "green", "green"])
        # rospy.sleep(2)
        # self._wheels_publisher.publish(WheelsCmdStamped(
        #     vel_left=self._vel_left, vel_right=self._vel_right))

        # self.turnWhite = True

    def callback_left(self, data):
        rospy.loginfo_once(f"Left encoder resolution: {data.resolution}")
        self._ticks_left = data.data
        if self._just_started_left:
            self._start_left = self._ticks_left
            self._just_started_left = False

    def callback_right(self, data):
        rospy.loginfo_once(f"Right encoder resolution: {data.resolution}")
        self._ticks_right = data.data
        if self._just_started_right:
            self._start_right = self._ticks_right
            self._just_started_right = False

    def compute_distance_straight(self):
        """
        This function calculates the linear distance traveled by the robot.
        It does this by taking the average of the tick differences from the left and right wheel encoders
        and multiplying the result by the distance per tick. This approach assumes the robot is moving
        straight or approximately straight.

        Formula:
        distance = ((Δ_left + Δ_right) / 2) * distance_per_tick
        where:
        - Δ_left = current_left_ticks - initial_left_ticks
        - Δ_right = current_right_ticks - initial_right_ticks
        """
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._start_left  # Change in left encoder ticks
            delta_right = self._ticks_right - self._start_right  # Change in right encoder ticks
            distance = (delta_left + delta_right) * \
                self._dist_per_tick / 2  # Average distance
            return distance
        return 1

    def compute_distance_rotated(self):
        """
        This function calculates the rotational distance (angular displacement) of the robot.
        It computes the difference in tick changes between the right and left wheels and multiplies
        it by the distance per tick. A positive result indicates clockwise rotation, and a negative
        result indicates counterclockwise rotation.

        Formula:
        distance = (Δ_right - Δ_left) * distance_per_tick
        where:
        - Δ_left = current_left_ticks - initial_left_ticks
        - Δ_right = current_right_ticks - initial_right_ticks
        """
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._start_left  # Change in left encoder ticks
            delta_right = self._ticks_right - self._start_right  # Change in right encoder ticks
            distance = (delta_right - delta_left) * \
                self._dist_per_tick  # Rotational distance
            return distance

    def compute_distance_curved(self):
        delta_right = self._ticks_right - self._start_right
        return delta_right * self._dist_per_tick

    def compute_distance_to_curve(self, radius=0.3, angle=pi/2):
        d_outer = angle * (radius + self._baseline / 2) * 1.3
        d_inner = angle * (radius - self._baseline / 2)
        return [d_outer, d_inner]

    def compute_velocities_curved(self, ratio_din_dout=1.0, right_turn=True):
        if right_turn:
            # Left outer, right inner arc
            vel_left = self._velocity
            vel_right = vel_left * ratio_din_dout
        else:
            # Right outer, left inner arc
            vel_right = self._velocity
            vel_left = vel_right * ratio_din_dout
        return vel_left, vel_right

    def drive(self, vel_left=0, vel_right=0, distance=0, section_id=SectionID.STRAIGHT):

        # Start driving with left and right wheel velocity
        self._start_left = self._ticks_left
        self._start_right = self._ticks_right
        rate = rospy.Rate(10)  # 10 Hz

        if section_id == SectionID.STRAIGHT:
            message = WheelsCmdStamped(
                vel_left=vel_left, vel_right=vel_right)
            self._wheels_publisher.publish(message)
            rospy.loginfo("Driving straight...")
            # Set light colour
            while abs(self.compute_distance_straight()) < distance:
                rospy.loginfo(
                    f"Distance traveled: {self.compute_distance_straight():.2f} m")
                self._rate.sleep()

        elif section_id == SectionID.CURVE:
            message = WheelsCmdStamped(
                vel_left=vel_left, vel_right=vel_right)
            self._wheels_publisher.publish(message)
            rospy.loginfo("Driving in a curve...")
            while self.compute_distance_curved() < distance:
                rospy.loginfo(
                    f"Distance turned: {self.compute_distance_curved():.2f} m")
                self._rate.sleep()

        rospy.signal_shutdown(
            "Target distance reached. Shutting down.")
        self._vel_left = self._vel_right = 0
        message = WheelsCmdStamped(
            vel_left=self._vel_left, vel_right=self._vel_right)
        self._wheels_publisher.publish(message)

    # def drive(self, vel_left=0, vel_right=0, distance=0, section_id=SectionID.STRAIGHT):
    #     self._start_left = self._ticks_left
    #     self._start_right = self._ticks_right

    #     # Set wheel velocities
    #     message = WheelsCmdStamped(vel_left=vel_left, vel_right=vel_right)
    #     self._wheels_publisher.publish(message)
    #     rospy.loginfo("Driving...")

    #     # Define the condition for completion based on the section type
    #     if section_id == SectionID.STRAIGHT:
    #         target_distance = distance
    #         distance_fn = self.compute_distance_straight
    #     elif section_id == SectionID.CURVE:
    #         target_distance = distance
    #         distance_fn = self.compute_distance_curved

    #     def check_distance(event):
    #         current_distance = distance_fn()
    #         rospy.loginfo(f"Distance traveled: {current_distance:.2f} m")

    #         if current_distance >= target_distance:
    #             self._vel_left = self._vel_right = 0
    #             self._wheels_publisher.publish(WheelsCmdStamped(
    #                 vel_left=self._vel_left, vel_right=self._vel_right))
    #             rospy.loginfo("Target distance reached. Stopping.")

    #             # Shutdown the node after completing the curve
    #             rospy.signal_shutdown(
    #                 "Target distance reached. Shutting down.")
    #             return True  # Stop the timer when the target distance is reached
    #         else:
    #             return False

    #     # Set up a timer to periodically check the distance without blocking
    #     rospy.Timer(rospy.Duration(0.1), check_distance)

    def rotate(self, angle=pi / 2, rot_cw=True):
        # Distance to rotate is just dr - dl but we can use the absolute value for fun
        distance_to_rotate = self._baseline / 2 * angle
        distance_to_rotate = distance_to_rotate  # FINE-TUNING
        # add your code here
        self._start_left = self._ticks_left
        self._start_right = self._ticks_right
        # 10 Hz
        if rot_cw:
            vel_right = self._velocity * -1 / 2
            vel_left = self._velocity / 2
        else:
            vel_right = self._velocity / 2
            vel_left = self._velocity * -1 / 2

        message = WheelsCmdStamped(
            vel_left=vel_left, vel_right=vel_right)
        self._wheels_publisher.publish(message)
        while abs(self.compute_distance_rotated()) < distance_to_rotate:
            rospy.loginfo(
                f"Distance rotated: {self.compute_distance_rotated():.2f} m")
            self._rate.sleep()

        self._vel_left = self._vel_right = 0
        message = WheelsCmdStamped(
            vel_left=self._vel_left, vel_right=self._vel_right)
        self._wheels_publisher.publish(message)

    def detect_color(self, hsvFrame, lower, upper):
        mask = cv2.inRange(hsvFrame, np.array(
            lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.dilate(mask, self.kernel)
        return mask

    def change_led_color(self, color_values):
        try:
            # Call the service with the desired color values
            _ = self.set_led_color_service(color_values)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    # Calculate the distance to point in pixels
    def get_distance_to_point(self, u, v):
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

        # Convert the image to HSV (OpenCV color space)
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Initialize an image to combine the detected colors
        output_image = image.copy()

        # Loop through the colors to detect them one by one
        for color, (lower, upper, _) in self.color_ranges.items():
            mask = self.detect_color(hsvFrame, lower, upper)

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # If contours are found, mark the detected color
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Only consider large contours
                if cv2.contourArea(largest_contour) > 1000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(output_image, (x, y),
                                  (x + w, y + h), (0, 0, 255), 2)

                    # Compute center of the bounding box
                    center_x, center_y = x + w // 2, y + h // 2

                    # Draw center point as a small circle
                    cv2.circle(output_image, (center_x, center_y),
                               5, (255, 0, 0), -1)  # Blue dot
                    _, _, distance = self.get_distance_to_point(
                        center_x, center_y)

                    rospy.loginfo(distance)
                    # Mark the detected color

                    self.distance = distance
                    self.detected_color = color
                    # Perform a specific task based on the detected color
                    if self.distance <= 0.1:
                        if self.turnWhite:
                            self.change_led_color(
                                ["white", "white", "white", "white", "white"])
                            self.turnWhite = False
                        # elif self.detected_color == "Red":
                        #     self.perform_task_red()
                        #     self.turnWhite = True
                        elif self.detected_color == "Green":
                            self.perform_task_green()
                            self.turnWhite = True
                        # elif self.detected_color == "Blue":
                        #     self.perform_task_blue()
                        #     self.turnWhite = True

        # Convert the image with rectangles back to CompressedImage
        image_message = self._bridge.cv2_to_compressed_imgmsg(output_image)

        # Publish the processed image with detected rectangles
        self.pub.publish(image_message)

    def perform_task_red(self):
        # Implement the task for when red color is detected
        rospy.loginfo("Red color detected! Performing task...")
        # Simulate task by sleeping for 2 seconds
        # Sleep to control processing rate and avoid message buildup
        # Perform the task

        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheels_publisher.publish(stop)
        self.change_led_color(["red", "red", "green", "red", "red"])

        rospy.sleep(3)

        self.drive(vel_left=self._velocity, vel_right=self._velocity,
                   distance=0.7, section_id=SectionID.STRAIGHT)

        # Set all lights to white after task
        # Sleep to control processing rate and avoid message buildup

    def perform_task_green(self):
        # Implement the task for when green color is detected
        rospy.loginfo("Green color detected! Performing task...")
        # Simulate task by sleeping for 2 seconds
        # Sleep to control processing rate and avoid message buildup
        # Perform the task

        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheels_publisher.publish(stop)

        rospy.sleep(3)

        colors = ["green", "white", "red", "green", "white"]
        self.change_led_color(colors)

        d_outer, d_inner = self.compute_distance_to_curve(
            radius=0.15, angle=pi / 2)
        vel_left, vel_right = self.compute_velocities_curved(
            ratio_din_dout=d_inner/d_outer, right_turn=False)
        self.drive(vel_left=vel_left, vel_right=vel_right,
                   distance=d_outer, section_id=SectionID.CURVE)

        # Set all lights to white after task
        # Sleep to control processing rate and avoid message buildup

    def perform_task_blue(self):
        # Implement the task for when blue color is detected
        rospy.loginfo("Blue color detected! Performing task...")
        # Simulate task by sleeping for 2 seconds
        # Sleep to control processing rate and avoid message buildup
        # Perform the task

        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheels_publisher.publish(stop)

        rospy.sleep(3)

        colors = ["white", "blue", "red", "white", "blue"]
        self.change_led_color(colors)

        d_outer, d_inner = self.compute_distance_to_curve(
            radius=0.15, angle=pi / 2)
        vel_left, vel_right = self.compute_velocities_curved(
            ratio_din_dout=d_inner/d_outer, right_turn=True)
        self.drive(vel_left=vel_left, vel_right=vel_right,
                   distance=d_outer, section_id=SectionID.CURVE)

    def on_shutdown(self):
        rospy.loginfo("Shutting down and stopping wheels.")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheels_publisher.publish(stop)
        rospy.sleep(1)

        # Unsubscribe from topics
        self.sub_left.unregister()
        self.sub_right.unregister()


if __name__ == '__main__':
    node = ColourDetectionNode(node_name='colour_detection_node')
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()
