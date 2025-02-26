#!/usr/bin/env python3

# import required libraries
from os import uname
from re import match
from math import pi
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from odometry.srv import SetLEDColor
from std_msgs.msg import ColorRGBA
# Assuming this service doesn't need any data returned
from enum import Enum


class SectionID(Enum):
    STRAIGHT = 1
    CURVE = 2


class MoveNode(DTROS):
    def __init__(self, node_name):
        super(MoveNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.color_dict = {
            "red": [1.0, 0.0, 0.0, 1.0],    # Red with full opacity
            "green": [0.0, 1.0, 0.0, 1.0],  # Green with full opacity
            "blue": [0.0, 0.0, 1.0, 1.0],   # Blue with full opacity
        }

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
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        # Wait for the service to be available before proceeding
        rospy.wait_for_service(f"/{self._vehicle_name}/set_led_color")
        self.set_led_color_service = rospy.ServiceProxy(
            f"/{self._vehicle_name}/set_led_color", SetLEDColor)

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
            self._publisher.publish(message)
            rospy.loginfo("Driving straight...")
            # Set light colour
            while abs(self.compute_distance_straight()) < distance:
                rospy.loginfo(
                    f"Distance traveled: {self.compute_distance_straight():.2f} m")
                self._rate.sleep()

        elif section_id == SectionID.CURVE:
            message = WheelsCmdStamped(
                vel_left=vel_left, vel_right=vel_right)
            self._publisher.publish(message)
            rospy.loginfo("Driving in a curve...")
            while self.compute_distance_curved() < distance:
                rospy.loginfo(
                    f"Distance turned: {self.compute_distance_curved():.2f} m")
                self._rate.sleep()

        self._vel_left = self._vel_right = 0
        message = WheelsCmdStamped(
            vel_left=self._vel_left, vel_right=self._vel_right)
        self._publisher.publish(message)

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
        self._publisher.publish(message)
        while abs(self.compute_distance_rotated()) < distance_to_rotate:
            rospy.loginfo(
                f"Distance rotated: {self.compute_distance_rotated():.2f} m")
            self._rate.sleep()

        self._vel_left = self._vel_right = 0
        message = WheelsCmdStamped(
            vel_left=self._vel_left, vel_right=self._vel_right)
        self._publisher.publish(message)

    def change_led_color(self, color_values):
        try:
            # Call the service with the desired color values
            _ = self.set_led_color_service(
                color_values[0], color_values[1], color_values[2], color_values[3])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def run(self):
        # rospy.sleep(2)  # Wait for initialization
        # self.change_led_color(self.color_dict["red"])
        # rospy.sleep(5)
        # self.change_led_color(self.color_dict["green"])

        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=1.1, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)
        # self.rotate(angle=pi/2, rot_cw=True)
        # rospy.sleep(1)

        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=0.7, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # d_outer, d_inner = self.compute_distance_to_curve(
        #     radius=0.15, angle=pi / 2)
        # vel_left, vel_right = self.compute_velocities_curved(
        #     ratio_din_dout=d_inner/d_outer, right_turn=True)
        # self.drive(vel_left=vel_left, vel_right=vel_right,
        #            distance=d_outer, section_id=SectionID.CURVE)
        # rospy.sleep(1)

        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=0.4, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # d_outer, d_inner = self.compute_distance_to_curve(
        #     radius=0.15, angle=pi / 2)
        # vel_left, vel_right = self.compute_velocities_curved(
        #     ratio_din_dout=d_inner/d_outer, right_turn=True)
        # self.drive(vel_left=vel_left, vel_right=vel_right,
        #            distance=d_outer, section_id=SectionID.CURVE)
        # rospy.sleep(1)

        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=0.7, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # self.rotate(angle=pi/2, rot_cw=True)
        # rospy.sleep(5)

        # self.change_led_color(self.color_dict["red"])
        # rospy.sleep(2)

        # Move straight for 0.5 meters
        rospy.sleep(2)
        self.change_led_color(self.color_dict["green"])
        self.drive(vel_left=self._velocity, vel_right=self._velocity,
                   distance=0.5, section_id=SectionID.STRAIGHT)
        rospy.sleep(1)

        # Rotate anti-clockwise 90 degrees
        self.rotate(angle=pi/2, rot_cw=False)
        rospy.sleep(1)

        # Drive in reverse direction for 0.3 meters
        self.drive(vel_left=-self._velocity, vel_right=-self._velocity,
                   distance=0.3, section_id=SectionID.STRAIGHT)
        rospy.sleep(1)

        self.change_led_color(self.color_dict["red"])
        rospy.sleep(2)

        # CODE FOR SQUARE
        # rospy.sleep(2)
        # self.change_led_color(self.color_dict["red"])
        # rospy.sleep(5)
        # self.change_led_color(self.color_dict["green"])

        # # First side of the square
        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=1.1, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # # First 90-degree clockwise rotation
        # self.rotate(angle=pi/2, rot_cw=True)
        # rospy.sleep(1)

        # # Second side of the square
        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=1.1, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # # Second 90-degree clockwise rotation
        # self.rotate(angle=pi/2, rot_cw=True)
        # rospy.sleep(1)

        # # Third side of the square
        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=1.1, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # # Third 90-degree clockwise rotation
        # self.rotate(angle=pi/2, rot_cw=True)
        # rospy.sleep(1)

        # # Fourth side of the square
        # self.drive(vel_left=self._velocity, vel_right=self._velocity,
        #            distance=1.1, section_id=SectionID.STRAIGHT)
        # rospy.sleep(1)

        # # Fourth 90-degree clockwise rotation to complete the square
        # self.rotate(angle=pi/2, rot_cw=True)
        rospy.sleep(5)

        # Change LED color to indicate the square path is complete
        self.change_led_color(self.color_dict["red"])
        rospy.sleep(2)

        # Explicitly call on_shutdown after exiting the loop
        self.on_shutdown()

    def on_shutdown(self):
        rospy.loginfo("Shutting down and stopping wheels.")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)
        rospy.sleep(1)

        # Unsubscribe from topics
        self.sub_left.unregister()
        self.sub_right.unregister()


if __name__ == '__main__':
    node = MoveNode(node_name='move_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()

    # keep spinning
    # rospy.spin()
