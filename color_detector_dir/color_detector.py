#!/usr/bin/env python3
import cv2
import numpy as np
from time import sleep
import os

# Get the number of splits from the environment variable
num_splits = int(os.environ.get('NUM_SPLITS', 2))  # Default to 2 splits if not set

print(f"Dividing the image into {num_splits} horizontal splits.")


def gst_pipeline_string():
    # Parameters from the camera_node
    # Refer here : https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/camera_driver/config/jetson_nano_camera_node/duckiebot.yaml
    res_w, res_h, fps = 640, 480, 30
    fov = 'full'
    # find best mode
    camera_mode = 3  # 
    # compile gst pipeline
    gst_pipeline = """ \
            nvarguscamerasrc \
            sensor-mode= exposuretimerange="100000 80000000" ! \
            video/x-raw(memory:NVMM), width=, height=, format=NV12, 
                framerate=/1 ! \
            nvjpegenc ! \
            appsink \
        """.format(
        camera_mode,
        res_w,
        res_h,
        fps
    )

    # ---
    print("Using GST pipeline: ``".format(gst_pipeline))
    return gst_pipeline


cap = cv2.VideoCapture()
cap.open(gst_pipeline_string(), cv2.CAP_GSTREAMER)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Put here your code!
    # You can now treat output as a normal numpy array
    # Do your magic here
    height, width, _ = frame.shape
    # Calculate the height of each split
    split_height = height // num_splits

    for i in range(num_splits):
        # Define the region of interest (ROI) for each split
        y_start = i * split_height
        y_end = (i + 1) * split_height if i < num_splits - 1 else height  # To handle the last split
        roi = frame[y_start:y_end, :]

        # Calculate the strongest RGB color in each split
        avg_color = np.mean(roi, axis=(0, 1))  # Mean of the pixels (BGR format)
        avg_color_rgb = avg_color[::-1]  # Convert to RGB from BGR

        print(f"Split {i + 1}: Strongest RGB color: {avg_color_rgb}")


    sleep(1)
