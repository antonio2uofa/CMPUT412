#!/bin/bash

# Prompt the user for the number of splits
echo "Enter the num of splits:"

# Read the input from the user
read num_splits

# Build the Docker image (if you haven't built it already)
echo "Building the Docker image..."
docker -H csc22927.local build -t colordetector .

# Run the Docker container with the -e flag, passing the number of splits as an environment variable
echo "Running the Docker container..."
docker -H csc22927.local run -e NUM_SPLITS=$num_splits -it --privileged -v /tmp/argus_socket:/tmp/argus_socket colordetector