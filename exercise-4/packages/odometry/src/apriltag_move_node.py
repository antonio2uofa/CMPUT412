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


class SectionID(Enum):
    STRAIGHT = 1
    CURVE = 2


class AprilTagMoveNode(DTROS):
    def __init__(self, node_name):
        super(AprilTagMoveNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        # Initialize parameters
        self._hostname = uname()[1]
        if not match(r"^csc\d+$", self._hostname):
            print("Pass in the hostname of the bot ![DUCKIEBOT]:")
            self._hostname = input()

        self._vehicle_name = self._hostname
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        # Encoder data and flags
        self._ticks_left = None
        self._ticks_right = None
        self._start_left = 0
        self._start_right = 0
        self._just_started_left = True
        self._just_started_right = True
        self._velocity = 0.25

        # Red line detection variables
        self._red_line_distance = float('inf')
        self._is_stopped_at_line = False
        self._stop_start_time = None
        self._stop_duration = 0  # In seconds

        # Recovery state - ignore red lines for a period after resuming
        self._in_recovery = False
        self._recovery_start_time = None
        self._recovery_distance_start = 0
        self._recovery_duration = 1.5  # Seconds to ignore red lines after stopping
        # Distance to move forward before detecting lines again
        self._recovery_distance = 3.0

        # Tag stop durations (in seconds)
        self._tag_stop_durations = {
            21: 3.0,    # Stop for 3 seconds for tag ID 21
            50: 2.0,    # Stop for 2 seconds for tag ID 50
            93: 1.0,    # Stop for 1 seconds for tag ID 93
            # Add more tag IDs and durations as needed
        }

        # Default tag ID if none is detected
        self._default_tag_id = 21  # Default to 3 seconds

        # Track the latest AprilTag ID
        self._latest_tag_id = self._default_tag_id

        # Subscribe to AprilTag topic to continuously track the latest ID
        self.apriltag_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/apriltag_detection_node/apriltag_id",
            Int16,
            self.apriltag_callback,
            queue_size=1)

        # Subscribe to encoder topics
        self.sub_left = rospy.Subscriber(
            self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(
            self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        # Subscribe to red line distance topic
        self.red_distance_sub = rospy.Subscriber(
            f"/{self._vehicle_name}/red_line_distance", Float32, self.red_distance_callback)

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

    def apriltag_callback(self, msg):
        """Callback to track the latest AprilTag ID"""
        self._latest_tag_id = msg.data
        rospy.loginfo_throttle(
            2.0, f"Tracking AprilTag ID: {self._latest_tag_id}")

    def compute_distance_straight(self):
        """Calculate linear distance traveled by the robot"""
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._start_left
            delta_right = self._ticks_right - self._start_right
            distance = (delta_left + delta_right) * self._dist_per_tick / 2
            return distance
        return 0

    def compute_distance_since_recovery(self):
        """Calculate distance traveled since recovery started"""
        if self._ticks_left is not None and self._ticks_right is not None:
            delta_left = self._ticks_left - self._recovery_distance_start
            delta_right = self._ticks_right - self._recovery_distance_start
            # Average the distance
            return (delta_left + delta_right) * self._dist_per_tick / 2
        return 0

    def stop_robot(self):
        """Stop the robot by setting wheel velocities to zero"""
        stop_msg = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_msg)
        rospy.loginfo("Robot stopped at red line")

    def start_forward_motion(self):
        """Start moving forward continuously"""
        forward_msg = WheelsCmdStamped(
            vel_left=self._velocity, vel_right=self._velocity)
        self._publisher.publish(forward_msg)
        rospy.loginfo("Robot moving forward")

    def handle_red_line_detection(self):
        """Handle red line detection and stopping logic"""
        # Only handle red line detection if we're not already stopped and not in recovery
        if not self._is_stopped_at_line and not self._in_recovery:
            # Stop the robot
            self.stop_robot()
            self._is_stopped_at_line = True
            self._stop_start_time = rospy.Time.now()

            # Use the tracked AprilTag ID - no need for blocking calls
            current_tag_id = self._latest_tag_id

            # Determine stop duration based on the retrieved AprilTag
            if current_tag_id in self._tag_stop_durations:
                self._stop_duration = self._tag_stop_durations[current_tag_id]
                rospy.loginfo(
                    f"Stopping for {self._stop_duration} seconds based on AprilTag {current_tag_id}")
            else:
                # Default stop duration if unknown ID
                self._stop_duration = 0.5
                rospy.loginfo(
                    f"Stopping for default duration of {self._stop_duration} seconds")

    def enter_recovery_mode(self):
        """Enter recovery mode after a stop to ignore red lines temporarily"""
        self._in_recovery = True
        self._recovery_start_time = rospy.Time.now()

        # Record tick position for distance tracking
        if self._ticks_left is not None:
            self._recovery_distance_start = self._ticks_left

        rospy.loginfo("Entered recovery mode - ignoring red lines temporarily")

        # Start moving forward
        self.start_forward_motion()

    def check_recovery_status(self):
        """Check if recovery period should end"""
        if not self._in_recovery:
            return

        # Check if we've moved far enough
        distance_traveled = self.compute_distance_since_recovery()
        elapsed_time = (rospy.Time.now() - self._recovery_start_time).to_sec()

        # End recovery if either time or distance condition is met
        if distance_traveled >= self._recovery_distance or elapsed_time >= self._recovery_duration:
            self._in_recovery = False
            rospy.loginfo(
                f"Exited recovery mode - traveled {distance_traveled:.2f}m in {elapsed_time:.2f}s")

    def timer_callback(self, event):
        """Timer callback for periodic checks"""
        # First, check if recovery period should end
        if self._in_recovery:
            self.check_recovery_status()
            return

        # If currently stopped, check if stop duration has elapsed
        if self._is_stopped_at_line:
            elapsed_time = (rospy.Time.now() - self._stop_start_time).to_sec()

            # Make sure we don't get stuck - add a safety timeout
            # 150% of intended time or 5 seconds, whichever is greater
            max_stop_time = max(self._stop_duration * 1.5, 5.0)

            if elapsed_time >= self._stop_duration or elapsed_time > max_stop_time:
                rospy.loginfo(
                    f"Stop duration completed ({elapsed_time:.2f}s). Entering recovery mode.")
                self._is_stopped_at_line = False
                self.enter_recovery_mode()

        # If not stopped or in recovery, check for red line
        elif self._red_line_distance < 0.1:
            self.handle_red_line_detection()

    def run(self):
        """Main run function with robot movement sequence"""
        rospy.loginfo("Starting AprilTagMoveNode execution...")

        # Wait a moment for initialization
        rospy.sleep(1)

        # Start by moving forward continuously
        self.start_forward_motion()

        # Main loop - we don't need to do much here since the callbacks handle the logic
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
        self.apriltag_sub.unregister()


if __name__ == '__main__':
    node = AprilTagMoveNode(node_name='move_node')
    rospy.on_shutdown(node.on_shutdown)
    node.run()
