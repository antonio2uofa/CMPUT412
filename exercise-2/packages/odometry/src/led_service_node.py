#!/usr/bin/env python3

# import required libraries
from os import uname
from re import match
import rospy
from duckietown_msgs.msg import LEDPattern
from odometry.srv import SetLEDColor, SetLEDColorResponse
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import ColorRGBA


class LedService(DTROS):
    def __init__(self, node_name):
        super(LedService, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Publisher to send the LED pattern message
        self.led_pub = rospy.Publisher(
            f"/{self._vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=10)

        # Define service to change LED color using the custom service
        self.set_led_service = rospy.Service(
            f"/{self._vehicle_name}/set_led_color", SetLEDColor, self.set_led_color_callback)

        rospy.loginfo("LED service is ready to set LED colors.")

    def set_led_color_callback(self, request):
        # Create the LEDPattern message
        led_pattern_msg = LEDPattern()
        led_pattern_msg.header.stamp = rospy.Time.now()

        # Set the color for the LED
        # Assuming that the `request` contains the list of colors (passed from the service call)
        # In your case, you would directly get these colors from the service request

        # Assign the list of ColorRGBA to the message
        color_1 = ColorRGBA(r=request.r, g=request.g,
                            b=request.b, a=request.a)  # Red color
        colors = [color_1] * 5
        led_pattern_msg.rgb_vals = colors
        led_pattern_msg.frequency = 0.0  # Static color (no flashing)

        # Publish the LED pattern message
        self.led_pub.publish(led_pattern_msg)
        rospy.loginfo("Set LEDs to the requested color pattern.")

        # Return an empty response (EmptyResponse is used because we're not returning any data)
        return SetLEDColorResponse(True)


if __name__ == "__main__":
    # Initialize the node with a unique name
    led_node = LedService(node_name="led_service_node")
    rospy.spin()
