#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun odometry bluered_detection_node.py

# wait for app to end
dt-launchfile-join