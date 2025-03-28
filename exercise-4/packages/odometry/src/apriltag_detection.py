#!/usr/bin/env python3

# potentially useful for part 1 of exercise 4

# import required libraries
import rospy
from os import uname
from re import match
import cv2
from cv_bridge import CvBridge
import dt_apriltags as apriltag
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LEDPattern
from odometry.srv import SetLEDColor
from std_msgs.msg import ColorRGBA, String, Int16


class ApriltagNode(DTROS):

    def __init__(self, node_name):
        super(ApriltagNode, self).__init__(
            node_name=node_name, node_type=NodeType.CONTROL)

        # Get hostname of vehicle
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname
        # Publisher to send the LED pattern message
        self.led_pub = rospy.Publisher(
            f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)

        # Initialize the LED dictionary
        self.led = {
            # Red with full opacity
            "red": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            # Green with full opacity
            "green": ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            # Blue with full opacity
            "blue": ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            # White with full opacity
            "white": ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
        }

        # initialize dt_apriltag detector
        self.detector = apriltag.Detector(families='tag36h11',
                                          nthreads=1,
                                          quad_decimate=1.0,
                                          quad_sigma=0.0,
                                          refine_edges=1,
                                          decode_sharpening=0.25,
                                          debug=0)

        # subscribe to camera feed
        self._bridge = CvBridge()

        # image processing topics
        self._undistort_topic = f"/{self._vehicle_name}/undistorted_node/image/compressed"
        self._apriltag_detection_topic = f"/{self._vehicle_name}/apriltag_detection_node/image/compressed"

        self.img_sub = rospy.Subscriber(
            self._undistort_topic, CompressedImage, self.camera_callback, queue_size=1)
        self.apriltag_pub = rospy.Publisher(
            self._apriltag_detection_topic, CompressedImage, queue_size=1)

        self.tag_id_pub = rospy.Publisher(
            f"/{self._vehicle_name}/apriltag_detection_node/apriltag_id", Int16, queue_size=2, latch=True)

        # movement topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        # Add parameter to control processing frequency
        # Process 5 frames per second by default
        self.processing_rate = rospy.get_param('~processing_rate', 5)
        self.latest_image_msg = None

    def change_led_color(self, cv):
        led_pattern_msg = LEDPattern()
        led_pattern_msg.header.stamp = rospy.Time.now()
        led_pattern_msg.rgb_vals = [
            self.led[cv[0]], self.led[cv[1]], self.led[cv[2]], self.led[cv[3]], self.led[cv[4]]]
        led_pattern_msg.frequency = 0.0  # Static color (no flashing)

        # Publish the LED pattern message
        self.led_pub.publish(led_pattern_msg)

    def process_image(self, image):

        # Convert to black and white
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray_img)

        # # Initialize an image to combine the detected colors
        # output_image = image.copy()

        # for r in results:
        #     # extract the bounding box (x, y)-coordinates for the AprilTag
        #     # and convert each of the (x, y)-coordinate pairs to integers
        #     (ptA, ptB, ptC, ptD) = r.corners
        #     ptB = (int(ptB[0]), int(ptB[1]))
        #     ptC = (int(ptC[0]), int(ptC[1]))
        #     ptD = (int(ptD[0]), int(ptD[1]))
        #     ptA = (int(ptA[0]), int(ptA[1]))

        #     # draw the bounding box of the AprilTag detection
        #     cv2.line(output_image, ptA, ptB, (0, 255, 0), 2)
        #     cv2.line(output_image, ptB, ptC, (0, 255, 0), 2)
        #     cv2.line(output_image, ptC, ptD, (0, 255, 0), 2)
        #     cv2.line(output_image, ptD, ptA, (0, 255, 0), 2)

        #     # draw the center (x, y)-coordinates of the AprilTag
        #     (cX, cY) = (int(r.center[0]), int(r.center[1]))

        #     tagID = str(r.tag_id)
        #     cv2.putText(output_image, tagID, (cX, cY),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return results, image

    def publish_augmented_img(self, image):
        # Convert the image with rectangles back to CompressedImage
        image_message = self._bridge.cv2_to_compressed_imgmsg(image)

        # Publish the processed image with detected rectangles
        self.apriltag_pub.publish(image_message)

        return

    def publish_leds(self, r):
        if len(r) == 0:
            self.change_led_color(
                ["white", "white", "blue", "white", "white"])
        else:
            for result in r:
                id = str(result.tag_id)
                if id == "21":
                    self.change_led_color(
                        ["red", "red", "green", "red", "red"])
                elif id == "50":
                    self.change_led_color(
                        ["blue", "blue", "green", "blue", "blue"])
                elif id == "93":
                    self.change_led_color(
                        ["green", "green", "blue", "green", "green"])
        return

    def camera_callback(self, msg):
        if rospy.is_shutdown():
            return

        # Store the most recent image
        self.latest_image_msg = msg

        return

    def process_loop(self):
        # Set up a rate object to control the processing frequency
        rate = rospy.Rate(self.processing_rate)  # Hz

        # Initialize storage for the latest image
        self.latest_image_msg = None

        while not rospy.is_shutdown():
            # Only process if we have received an image
            if self.latest_image_msg is not None:
                # Convert the compressed image to an OpenCV image
                image = self._bridge.compressed_imgmsg_to_cv2(
                    self.latest_image_msg)

                # Process the image
                results, output_img = self.process_image(image)
                # self.publish_augmented_img(output_img)

                if len(results) > 0:
                    id_msg = Int16()
                    id_msg.data = int(results[0].tag_id)
                    self.tag_id_pub.publish(id_msg)

                self.publish_leds(results)

            # Sleep to maintain the desired rate
            rate.sleep()


if __name__ == '__main__':
    # create the node
    node = ApriltagNode(node_name='apriltag_detector_node')

    # Start the processing loop in a separate thread
    import threading
    processing_thread = threading.Thread(target=node.process_loop)
    processing_thread.daemon = True
    processing_thread.start()

    # Keep the node running
    rospy.spin()
