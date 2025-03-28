#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun odometry move_node.py

# wait for app to end
dt-launchfile-join