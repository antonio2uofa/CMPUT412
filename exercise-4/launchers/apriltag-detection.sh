#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun odometry apriltag_detection.py

# wait for app to end
dt-launchfile-join