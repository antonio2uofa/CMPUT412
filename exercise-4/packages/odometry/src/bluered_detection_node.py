#!/usr/bin/env python3

# import required libraries
from os import uname
from re import match
from math import pi
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Float32, Int16
from enum import Enum


class SequenceState(Enum):
    LOOKING_FOR_BLUE_LINES = 1
    APPROACHING_BLUE_LINE = 2
    WAITING_FOR_RED_LINE = 3
    STOPPED_AT_RED_LINE = 4
    WAITING_AFTER_SEQUENCE = 5


class BlueRedSequenceNode(DTROS):
    def __init__(self, node_name):
        super(BlueRedSequenceNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname

        # Encoder data and flags
        self._ticks_left = None
        self._ticks_right = None
        self._start_left = 0
        self._start_right = 0
        self._just_started_left = True
        self._just_started_right = True
        self._velocity = 0.20

        # Line detection variables
        self._red_line_distance = float('inf')
        self._blue_line_double_distance = float(
            'inf')  # Distance to double blue lines
        # Distance to nearest blue line
        self._blue_line_distance = float('inf')

        # State tracking variables
        self._current_state = SequenceState.LOOKING_FOR_BLUE_LINES
        self._previous_state = None  # To track state transitions
        self._sequence_start_time = None
        self._stop_start_time = None
        # Wait 1 second after red line disappears
        self._wait_after_sequence_duration = 1.0

        # Debug flag for enhanced logging
        self._debug = True

        # Log initial state
        rospy.loginfo(f"INITIAL STATE: {self._current_state.name}")

        # Subscribe to encoder topics
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Subscribe to line detection topics
        self.red_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/red_line_distance", Float32, self.red_distance_callback)

        # Subscribe to blue line detection topics
        self.blue_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/blue_line_distance", Float32, self.blue_distance_callback)

        self.blue_double_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/blue_line_double_distance", Float32, self.blue_double_distance_callback)

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

        # Create a timer for periodic checks
        self._timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # Publisher for wheel commands
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

    def change_state(self, new_state):
        """Helper method to change state with logging"""
        if self._current_state != new_state:
            self._previous_state = self._current_state
            self._current_state = new_state
            rospy.loginfo(
                f"STATE TRANSITION: {self._previous_state.name} -> {self._current_state.name}")
            return True
        return False

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

    def red_distance_callback(self, msg):
        """Callback for red line distance"""
        self._red_line_distance = msg.data
        if self._debug and self._red_line_distance < 0.2:
            rospy.loginfo_throttle(
                1.0, f"Red line distance: {self._red_line_distance:.3f}m")

    def blue_distance_callback(self, msg):
        """Callback for single blue line distance"""
        self._blue_line_distance = msg.data
        if self._debug and self._blue_line_distance < 0.2:
            rospy.loginfo_throttle(
                1.0, f"Blue line distance: {self._blue_line_distance:.3f}m")

    def blue_double_distance_callback(self, msg):
        """Callback for double blue lines distance"""
        self._blue_line_double_distance = msg.data
        if self._debug and self._blue_line_double_distance < float('inf'):
            rospy.loginfo_throttle(
                1.0, f"Double blue line distance: {self._blue_line_double_distance:.3f}m")

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
        # rospy.loginfo("Robot moving forward")

    def handle_state_machine(self):
        """Main state machine for the sequence detection"""
        # Get current state
        state = self._current_state

        # State: Looking for blue lines
        if state == SequenceState.LOOKING_FOR_BLUE_LINES:
            # Move forward while looking for blue lines
            self.start_forward_motion()

            # Check if we can see the double blue lines
            if self._blue_line_double_distance < float('inf'):
                rospy.loginfo(
                    f"Detected double blue lines at distance {self._blue_line_double_distance:.2f}m")
                self.change_state(SequenceState.APPROACHING_BLUE_LINE)

        # State: Approaching blue lines
        elif state == SequenceState.APPROACHING_BLUE_LINE:
            # Continue moving forward but check if we're close to the nearest blue line
            if self._blue_line_distance < 0.1:  # Stop when close to the first blue line
                self.stop_robot()
                rospy.loginfo(
                    "Stopped at the first blue line. Now waiting for red line detection.")
                self.change_state(SequenceState.WAITING_FOR_RED_LINE)
                self._sequence_start_time = rospy.Time.now()

        # State: Waiting for red line
        elif state == SequenceState.WAITING_FOR_RED_LINE:
            # Check if we see a red line
            if self._red_line_distance < 0.1:
                rospy.loginfo(
                    f"Detected red line while waiting. Distance: {self._red_line_distance:.2f}m")
                self.change_state(SequenceState.STOPPED_AT_RED_LINE)
                self._stop_start_time = rospy.Time.now()

            # Timeout if we've been waiting too long (optional safety feature)
            elapsed_time = (rospy.Time.now() -
                            self._sequence_start_time).to_sec()
            if elapsed_time > 10.0:  # 10 second timeout
                rospy.logwarn(
                    "Timeout waiting for red line. Restarting sequence.")
                self.change_state(SequenceState.LOOKING_FOR_BLUE_LINES)
                self.start_forward_motion()

        # State: Stopped at red line
        elif state == SequenceState.STOPPED_AT_RED_LINE:
            # Wait until the red line is no longer detected (maybe we've moved past it or it disappeared)
            if self._red_line_distance > 0.1:
                rospy.loginfo(
                    "Red line no longer detected. Starting wait period.")
                self.change_state(SequenceState.WAITING_AFTER_SEQUENCE)
                self._stop_start_time = rospy.Time.now()

        # State: Waiting after sequence
        elif state == SequenceState.WAITING_AFTER_SEQUENCE:
            # Wait for the specified duration after the red line
            elapsed_time = (rospy.Time.now() - self._stop_start_time).to_sec()
            if elapsed_time >= self._wait_after_sequence_duration:
                rospy.loginfo(
                    f"Completed {self._wait_after_sequence_duration}s wait. Restarting sequence.")
                self.change_state(SequenceState.LOOKING_FOR_BLUE_LINES)
                self.start_forward_motion()

    def timer_callback(self, event):
        """Timer callback for periodic checks of the state machine"""
        self.handle_state_machine()

        # Periodically log the current state for easier debugging
        if self._debug:
            rospy.loginfo_throttle(
                5.0, f"CURRENT STATE: {self._current_state.name}")

            # Log the line distances
            rospy.loginfo_throttle(5.0,
                                   f"Distance metrics - Red: {self._red_line_distance:.2f}m, " +
                                   f"Blue: {self._blue_line_distance:.2f}m, " +
                                   f"Double Blue: {self._blue_line_double_distance:.2f}m")

    def run(self):
        """Main run function"""
        rospy.loginfo("Starting Blue and Red Line Sequence Detector...")

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
        self.red_distance_sub.unregister()
        self.blue_distance_sub.unregister()
        self.blue_double_distance_sub.unregister()


if __name__ == '__main__':
    node = BlueRedSequenceNode(node_name='blue_red_sequence_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()
