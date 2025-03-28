#!/usr/bin/env python3

import rospy
import numpy as np
from os import uname
from re import match
from math import pi
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
import cv2
from cv_bridge import CvBridge


class SafeNavigationNode(DTROS):
    def __init__(self, node_name):
        super(SafeNavigationNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Image processing topics
        self._undistort_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"
        self._blue_detection_topic = f"/{self._vehicle_name}/blue_shape_detection_node/image/compressed"
        self._homography_param = f"/{self._vehicle_name}/updated_extrinsics"

        # Encoder data and flags
        self._ticks_left = None
        self._ticks_right = None
        self._start_left = 0
        self._start_right = 0
        self._just_started_left = True
        self._just_started_right = True
        self._velocity = 0.25

        # Blue shape detection variables
        self._blue_shape_distance = float('inf')
        self._blue_shape_center_x = None
        self._blue_shape_center_y = None
        self._target_distance = 0.5  # Stop 1.5 units from blue shape
        self._has_found_blue_shape = False
        self._sequence_started = False

        # Debug flag
        self._debug = True

        # Subscribe to encoder topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Subscribe to camera feed for blue detection
        self._bridge = CvBridge()
        self.camera_sub = rospy.Subscriber(
            self._undistort_topic, CompressedImage, self.camera_callback, queue_size=1)

        # Publisher for visualization
        self.img_pub = rospy.Publisher(
            self._blue_detection_topic, CompressedImage, queue_size=1)

        # Get parameters
        self._radius = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)  # Default radius
        self._resolution_left = rospy.get_param(
            f'/{self._vehicle_name}/left_wheel_encoder_node/resolution', 135)
        self._resolution_right = rospy.get_param(
            f'/{self._vehicle_name}/right_wheel_encoder_node/resolution', 135)
        self._baseline = rospy.get_param(
            f'/{self._vehicle_name}/kinematics_node/baseline', 0.1)

        if self._resolution_right != self._resolution_left:
            rospy.logwarn(
                "The resolutions of the left and right wheels do not match!")

        self._dist_per_tick = self._radius * 2 * pi / self._resolution_left

        # Initialize homography matrix
        self._homography = np.array(rospy.get_param(
            self._homography_param, default=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])).reshape(3, 3)

        # Blue color range in HSV
        self.blue_lower = (100, 100, 100)
        self.blue_upper = (140, 255, 255)

        # Minimum area for valid blue shape detection
        self.min_contour_area = 1000  # Larger threshold for a substantial shape

        # Kernel for morphological operations
        self.kernel = np.ones((3, 3), "uint8")

        # Create a timer for checking proximity to blue shape
        self._timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # Publisher for wheel commands
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        rospy.loginfo("Blue Shape Navigation Node initialized")

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

    def detect_blue(self, hsvFrame):
        """Detect blue color in HSV image"""
        mask = cv2.inRange(hsvFrame, np.array(
            self.blue_lower, dtype=np.uint8), np.array(self.blue_upper, dtype=np.uint8))
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

        # Ensure we return float values
        if isinstance(euclidean_dist, np.ndarray):
            euclidean_dist = float(euclidean_dist)

        return float(X), float(Y), euclidean_dist

    def camera_callback(self, msg):
        """Process camera image to detect large blue shape"""
        if rospy.is_shutdown():
            return

        # Convert the compressed image to an OpenCV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # Get image dimensions
        height, width = image.shape[:2]
        image_center_x = width // 2
        image_center_y = height // 2

        # Convert the image to HSV
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a copy for visualization
        output_image = image.copy()

        # Detect blue
        mask = self.detect_blue(hsvFrame)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize detection flags
        found_central_blue = False

        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(
            c) > self.min_contour_area]

        if valid_contours:
            # Sort by area (largest first)
            valid_contours.sort(key=cv2.contourArea, reverse=True)

            # Get the largest contour
            largest_contour = valid_contours[0]

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2

            # Check if it's close to the center of the image (within 25% of the center)
            center_threshold_x = width * 0.25
            center_threshold_y = height * 0.25

            if (abs(center_x - image_center_x) < center_threshold_x and
                    abs(center_y - image_center_y) < center_threshold_y):

                found_central_blue = True
                self._blue_shape_center_x = center_x
                self._blue_shape_center_y = center_y

                # Calculate distance
                _, _, distance = self.get_distance_to_point(center_x, center_y)
                self._blue_shape_distance = distance

                # Draw the detection
                cv2.rectangle(output_image, (x, y),
                              (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(output_image, (center_x, center_y),
                           5, (0, 255, 255), -1)

                # Add distance text
                cv2.putText(output_image, f"Dist: {distance:.3f}m",
                            (center_x - 70, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Log detection
                if self._debug:
                    rospy.loginfo_throttle(1.0,
                                           f"Detected central blue shape at ({center_x}, {center_y}) with distance {distance:.3f}m")

        # If no central blue shape found, reset distance
        if not found_central_blue:
            self._blue_shape_distance = float('inf')
            self._blue_shape_center_x = None
            self._blue_shape_center_y = None

        # Publish the visualization image
        image_message = self._bridge.cv2_to_compressed_imgmsg(output_image)
        self.img_pub.publish(image_message)

    def compute_distance_straight(self):
        """Calculate linear distance traveled by the robot"""
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._start_left
            delta_right = self._ticks_right - self._start_right
            distance = (delta_left + delta_right) * self._dist_per_tick / 2
            return distance
        return 0

    def drive_distance(self, distance):
        """Drive the robot forward for the specified distance"""
        # Reset encoder start positions
        self._start_left = self._ticks_left
        self._start_right = self._ticks_right

        # Start moving forward
        self.start_forward_motion()

        # Wait until we've traveled the desired distance
        while self.compute_distance_straight() < distance:
            rospy.sleep(0.05)

        # Stop when we've reached the target distance
        self.stop_robot()
        rospy.loginfo(f"Completed {distance}m forward movement")

    def stop_robot(self):
        """Stop the robot by setting wheel velocities to zero"""
        stop_msg = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_msg)
        rospy.loginfo("Robot stopped")

    def start_forward_motion(self):
        """Start moving forward continuously"""
        forward_msg = WheelsCmdStamped(
            vel_left=self._velocity, vel_right=self._velocity)
        self._publisher.publish(forward_msg)
        rospy.loginfo("Robot moving forward")

    def rotate(self, angle=pi/4, rot_cw=True):
        """Rotate the robot by the specified angle"""
        # Reset encoder start positions
        self._start_left = self._ticks_left
        self._start_right = self._ticks_right

        # Calculate distance to rotate
        distance_to_rotate = self._baseline / 2 * angle

        # Set rotation velocities
        if rot_cw:
            vel_right = self._velocity * -1 / 2
            vel_left = self._velocity / 2
            rospy.loginfo(
                f"Rotating CW by {angle:.2f} radians ({angle * 180 / pi:.1f} degrees)")
        else:
            vel_right = self._velocity / 2
            vel_left = self._velocity * -1 / 2
            rospy.loginfo(
                f"Rotating CCW by {angle:.2f} radians ({angle * 180 / pi:.1f} degrees)")

        # Send rotation command
        rotation_msg = WheelsCmdStamped(
            vel_left=vel_left, vel_right=vel_right)
        self._publisher.publish(rotation_msg)

        # Wait until we've rotated the desired amount
        while abs(self.compute_distance_rotated()) < distance_to_rotate:
            rospy.sleep(0.05)

        # Stop when we've reached the target rotation
        self.stop_robot()
        if rot_cw:
            rospy.loginfo(
                f"Completed {angle * 180 / pi:.1f} degree CW rotation")
        else:
            rospy.loginfo(
                f"Completed {angle * 180 / pi:.1f} degree CCW rotation")

    def compute_distance_rotated(self):
        """Calculate rotational distance of the robot"""
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._start_left
            delta_right = self._ticks_right - self._start_right
            distance = (delta_right - delta_left) * self._dist_per_tick
            return distance
        return 0

    def execute_movement_sequence(self):
        """Execute the predefined movement sequence"""
        rospy.loginfo("Starting movement sequence")

        # First, wait a moment
        rospy.sleep(3.0)

        # 1. Rotate 45 degrees clockwise
        self.rotate(angle=pi/4, rot_cw=True)
        rospy.sleep(0.5)

        # 2. Drive forward 0.5m
        self.drive_distance(0.5)
        rospy.sleep(0.5)

        # 3. Rotate 45 degrees counter-clockwise (back to straight)
        self.rotate(angle=pi/4, rot_cw=False)
        rospy.sleep(0.5)

        # 4. Drive forward 0.75m
        self.drive_distance(0.3)
        rospy.sleep(0.5)

        # 5. Rotate 45 degrees counter-clockwise
        self.rotate(angle=pi/4, rot_cw=False)
        rospy.sleep(0.5)

        # 6. Drive forward 0.5m
        self.drive_distance(0.2)
        rospy.sleep(0.5)

        # 7. Rotate 45 degrees clockwise (back to straight)
        self.rotate(angle=pi/4, rot_cw=True)
        rospy.sleep(0.5)

        # 8. Drive forward 0.5m
        self.drive_distance(0.1)
        rospy.sleep(0.5)

        rospy.loginfo("Movement sequence completed!")

    def timer_callback(self, event):
        """Timer callback to check proximity to blue shape and initiate sequence"""
        # If we've already started the sequence, don't do anything
        if self._sequence_started:
            return

        # If we haven't found the blue shape yet, keep moving forward to find it
        if not self._has_found_blue_shape and self._blue_shape_distance < float('inf'):
            self._has_found_blue_shape = True
            rospy.loginfo(
                f"Detected blue shape at distance {self._blue_shape_distance:.2f}m")

        # If we found the blue shape, approach until we're at the target distance
        if self._has_found_blue_shape and not self._sequence_started:
            if self._blue_shape_distance <= self._target_distance:
                self.stop_robot()
                rospy.loginfo(
                    f"Arrived at target distance: {self._blue_shape_distance:.2f}m")
                self._sequence_started = True
                # Start movement sequence in a separate thread to not block the callbacks
                import threading
                threading.Thread(target=self.execute_movement_sequence).start()
            else:
                # Keep moving forward
                self.start_forward_motion()

    def run(self):
        """Main run function"""
        rospy.loginfo("Starting Blue Shape Navigation...")

        # Wait a moment for initialization
        rospy.sleep(1)

        # Start by moving forward
        self.start_forward_motion()

        # Main loop is handled by the timer callback
        rospy.spin()

    def on_shutdown(self):
        """Handle shutdown procedure"""
        # Stop the timer
        if self._timer:
            self._timer.shutdown()

        rospy.loginfo("Shutting down and stopping wheels.")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)
        rospy.sleep(0.5)

        # Unsubscribe from topics
        self.sub_left.unregister()
        self.sub_right.unregister()
        self.camera_sub.unregister()


if __name__ == '__main__':
    node = SafeNavigationNode(node_name='safe_navigation_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()
