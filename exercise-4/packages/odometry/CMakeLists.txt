cmake_minimum_required(VERSION 2.8.3)
project(odometry)

# Add the required components for your package
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation  # Needed for generating services
)

# Add service files (you can list other services here as well)
add_service_files(
  FILES
  SetLEDColor.srv  # Replace with your service file
)

# Generate messages and services
generate_messages(
  DEPENDENCIES
  std_msgs  # Add any other dependencies if necessary
)

# Declare the catkin package
catkin_package()

