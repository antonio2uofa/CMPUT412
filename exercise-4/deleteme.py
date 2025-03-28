# SLOPE = -0.75

import cv2
import numpy as np
import pyautogui  # To get screen size

color_ranges = {
    "Yellow": ((25, 30, 50), (40, 255, 255), (0, 255, 255)),
    "White": ((100, 0, 200), (255, 255, 255), (255, 255, 255)),
}

# Load the image
image_path = "/home/antonio/Pictures/Screenshots/curve_far.png"
imageFrame = cv2.imread(image_path)

if imageFrame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Get screen size
screen_width, screen_height = pyautogui.size()

# Resize the image to 20% of the screen size
new_width = int(screen_width * 0.2)
new_height = int(screen_height * 0.2)
imageFrame = cv2.resize(imageFrame, (new_width, new_height))

# Convert image to HSV
hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# Morphological Transform, Dilation for the yellow mask to smooth them
kernel = np.ones((5, 5), "uint8")

# Function to find the leftmost edge or best-fit line of the largest white polygon

# Process Yellow and White colors
white_centroid = None
yellow_centroids = []
best_centroid, best_yellow_centroid = None, None
best_contour, best_yellow_contour = None, None
leftmost, line_start, line_end = 0, 0, 0
leftmost_yellow, line_start_yellow, line_end_yellow = 0, 0, 0


for color_name in ["Yellow", "White"]:
    lower = np.array(color_ranges[color_name][0], np.uint8)
    upper = np.array(color_ranges[color_name][1], np.uint8)
    mask = cv2.inRange(hsvFrame, lower, upper)
    mask = cv2.dilate(mask, kernel)

    max_area = 0
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 3:
        for contour in contours:
            area = cv2.contourArea(contour)
            if color_name == "Yellow":
                if area > 300:  # Only consider larger contours
                    if area > max_area:
                        max_area = area
                        best_yellow_contour = contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    # Draw the polygon
                    # cv2.drawContours(
                    #     imageFrame, [approx], -1, color_ranges[color_name][2], 2)
                    # Compute the centroid of the polygon
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        yellow_centroids.append((cX, cY))  # Store centroid

            elif color_name == "White":
                if area > 300 and area > max_area:
                    max_area = area
                    best_contour = contour

            if best_contour is not None:
                # Get the leftmost point (smallest X value)
                leftmost = tuple(
                    best_contour[best_contour[:, :, 0].argmin()][0])

                # Fit a line through the contour points
                [vx, vy, x0, y0] = cv2.fitLine(
                    best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                line_start = (int(x0 - vx * 1000), int(y0 - vy * 1000))
                line_end = (int(x0 + vx * 1000), int(y0 + vy * 1000))

    else:
        for contour in contours:
            area = cv2.contourArea(contour)
            if color_name == "Yellow":
                if area > 300 and area > max_area:
                    max_area = area
                    best_yellow_contour = contour

            elif color_name == "White":
                if area > 300 and area > max_area:
                    max_area = area
                    best_contour = contour

            if best_contour is not None:
                # Get the leftmost point (smallest X value)
                leftmost = tuple(
                    best_contour[best_contour[:, :, 0].argmin()][0])

                # Fit a line through the contour points
                [vx, vy, x0, y0] = cv2.fitLine(
                    best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                line_start = (int(x0 - vx * 1000), int(y0 - vy * 1000))
                line_end = (int(x0 + vx * 1000), int(y0 + vy * 1000))

    if line_start and line_end:
        cv2.line(imageFrame, line_start, line_end,
                 (255, 0, 255), 2)  # White guiding lane

    if line_start_yellow and line_end_yellow:
        cv2.line(imageFrame, line_start_yellow, line_end_yellow,
                 (255, 0, 255), 2)  # White guiding lane

    # Sort centroids by distance to bottom-left corner (0, height)
    yellow_centroids.sort(key=lambda c: np.sqrt(
        c[0]**2 + (imageFrame.shape[0] - c[1])**2))

    if len(yellow_centroids) >= 3:
        # First and third centroids
        centroid1, centroid3 = yellow_centroids[:3:2]

        # Fit a line through the two centroids
        yellow_points = np.array([centroid1, centroid3], dtype=np.int32)
        [vx, vy, x0, y0] = cv2.fitLine(
            yellow_points, cv2.DIST_L2, 0, 0.01, 0.01)

        # Extend the yellow line
        line_start_yellow = (int(x0 - vx * 1000), int(y0 - vy * 1000))
        line_end_yellow = (int(x0 + vx * 1000), int(y0 + vy * 1000))

        # Draw the yellow lane (Blue)
        cv2.line(imageFrame, line_start_yellow,
                 line_end_yellow, (255, 0, 0), 2)
    elif best_yellow_contour is not None:
        # Get the leftmost point (smallest X value)
        leftmost_yellow = tuple(
            best_yellow_contour[best_yellow_contour[:, :, 0].argmin()][0])

        # Fit a line through the contour points
        [vx, vy, x0, y0] = cv2.fitLine(
            best_yellow_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        line_start_yellow = (int(x0 - vx * 1000), int(y0 - vy * 1000))
        line_end_yellow = (int(x0 + vx * 1000), int(y0 + vy * 1000))


# Compute the midpoint line (averaging corresponding start and end points)

mid_start = ((line_start[0] + line_start_yellow[0]) // 2,
             (line_start[1] + line_start_yellow[1]) // 2)
mid_end = ((line_end[0] + line_end_yellow[0]) // 2,
           (line_end[1] + line_end_yellow[1]) // 2)

# Draw the original lane lines
cv2.line(imageFrame, line_start, line_end,
         (255, 0, 255), 2)  # White lane (Magenta)
cv2.line(imageFrame, line_start_yellow, line_end_yellow,
         (255, 0, 0), 2)  # Yellow lane (Blue)

# Draw the vertical centerline
cv2.line(imageFrame, mid_start, mid_end, (0, 255, 0), 2)  # Centerline (Green)

cv2.imshow("Lane Following Guide", imageFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# color_ranges = {
#     "Yellow": ((25, 30, 50), (40, 255, 255), (0, 255, 255)),
#     "White": ((100, 0, 200), (255, 255, 255), (255, 255, 255)),
# }

# # Load the image
# image_path = "/home/antonio/Pictures/Screenshots/right_side.png"
# imageFrame = cv2.imread(image_path)

# if imageFrame is None:
#     print(f"Error: Could not load image from {image_path}")
#     exit()

# # Get screen size
# screen_width, screen_height = pyautogui.size()

# # Resize the image to 20% of the screen size
# new_width = int(screen_width * 0.2)
# new_height = int(screen_height * 0.2)
# imageFrame = cv2.resize(imageFrame, (new_width, new_height))

# # Convert image to HSV
# hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# # Define a kernel for morphological operations
# kernel = np.ones((5, 5), "uint8")

# # Function to detect polygons and draw them


# def draw_polygons(mask, color):
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 500:  # Only consider larger contours
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#             # Draw the polygon
#             cv2.drawContours(imageFrame, [approx], -1, color, 2)

# # Function to get centroids of the largest contour


# def get_largest_centroid(mask):
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     max_area = 0
#     best_centroid = None

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 500 and area > max_area:
#             max_area = area
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 best_centroid = (cX, cY)

#     return best_centroid

# # Function to find the leftmost edge or best-fit line of the largest white polygon


# def get_white_lane_line(mask):
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     max_area = 0
#     best_contour = None

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 500 and area > max_area:
#             max_area = area
#             best_contour = contour

#     if best_contour is not None:
#         # Get the leftmost point (smallest X value)
#         leftmost = tuple(best_contour[best_contour[:, :, 0].argmin()][0])

#         # Fit a line through the contour points
#         [vx, vy, x0, y0] = cv2.fitLine(
#             best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
#         line_start = (int(x0 - vx * 1000), int(y0 - vy * 1000))
#         line_end = (int(x0 + vx * 1000), int(y0 + vy * 1000))

#         return leftmost, line_start, line_end

#     return None, None, None


# # Process Yellow and White colors
# white_centroid = None
# yellow_centroids = []

# for color_name in ["Yellow", "White"]:
#     lower = np.array(color_ranges[color_name][0], np.uint8)
#     upper = np.array(color_ranges[color_name][1], np.uint8)
#     mask = cv2.inRange(hsvFrame, lower, upper)
#     mask = cv2.dilate(mask, kernel)

#     draw_polygons(mask, color_ranges[color_name][2])  # Draw detected polygons

#     if color_name == "Yellow":
#         centroid = get_largest_centroid(mask)  # Get centroid for Yellow
#         if centroid:
#             yellow_centroids.append(centroid)
#     elif color_name == "White":
#         white_centroid = get_largest_centroid(mask)  # Get centroid for White
#         leftmost_white, white_line_start, white_line_end = get_white_lane_line(
#             mask)

# # Draw guiding path for the Yellow lane
# if len(yellow_centroids) >= 2:
#     c1, c3 = yellow_centroids[0], yellow_centroids[-1]
#     cv2.line(imageFrame, c1, c3, (0, 255, 255), 2)  # Yellow path

# # Draw White lane guide (either leftmost edge or best-fit line)
# if white_centroid:
#     cv2.circle(imageFrame, white_centroid, 5,
#                (255, 255, 255), -1)  # Mark centroid
# if white_line_start and white_line_end:
#     cv2.line(imageFrame, white_line_start, white_line_end,
#              (255, 255, 255), 2)  # White guiding lane

# # Show the final image
# cv2.imshow("Lane Following Guide", imageFrame)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# color_ranges = {
#     "Red": ((0, 50, 50), (20, 255, 255), (0, 0, 255)),
#     "Green": ((35, 60, 149), (85, 255, 255), (0, 255, 0)),
#     "Blue": ((100, 100, 100), (140, 255, 255), (255, 0, 0)),
#     "Yellow": ((25, 50, 50), (40, 255, 255), (0, 255, 255)),
#     "White": ((0, 0, 250), (180, 50, 255), (255, 255, 255)),
# }

# # Load the image
# image_path = "/home/antonio/Pictures/Screenshots/right_side.png"
# imageFrame = cv2.imread(image_path)

# if imageFrame is None:
#     print(f"Error: Could not load image from {image_path}")
#     exit()

# # Get screen size
# screen_width, screen_height = pyautogui.size()

# # Calculate the new dimensions (20% of the screen size)
# new_width = int(screen_width * 0.2)
# new_height = int(screen_height * 0.2)

# # Resize the image to 20% of the screen size
# imageFrame = cv2.resize(imageFrame, (new_width, new_height))

# # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
# hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# # Set range for yellow color and define the mask
# yellow_lower = np.array(color_ranges["Yellow"][0], np.uint8)
# yellow_upper = np.array(color_ranges["Yellow"][1], np.uint8)
# yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

# # Morphological Transform, Dilation for the yellow mask to smooth them
# kernel = np.ones((5, 5), "uint8")
# yellow_mask = cv2.dilate(yellow_mask, kernel)

# # Function to draw yellow lines and centroids


# def draw_yellow_lines(mask, color):
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     centroids = []  # To store the centroids of yellow polygons

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 300:  # Only consider larger contours
#             # Polygon Approximation
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)

#             # Compute the centroid of the polygon
#             M = cv2.moments(approx)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 centroids.append((cX, cY))  # Store centroid

#     return centroids


# # Draw yellow lines and centroids
# centroids = draw_yellow_lines(yellow_mask, (0, 255, 255))  # Yellow in BGR

# # Sort centroids by distance to bottom-left corner (0, height)
# centroids.sort(key=lambda c: np.sqrt(
#     c[0]**2 + (imageFrame.shape[0] - c[1])**2))

# # Take the first, second, and third centroids
# centroid1, centroid2, centroid3 = centroids[:3]

# # # Draw the line between the first and second centroids (Color: Green)
# cv2.line(imageFrame, centroid1, centroid2, (0, 255, 0), 2)

# # Draw the line between the first and third centroids (Color: Blue)
# cv2.line(imageFrame, centroid1, centroid3, (255, 0, 0), 2)

# # Compute the vector from centroid1 to centroid2
# dx = centroid2[0] - centroid1[0]
# dy = centroid2[1] - centroid1[1]

# # Compute the normal (perpendicular) vector for the first and second centroids
# normal_x = -dy
# normal_y = dx

# # Normalize the normal vector
# length = np.sqrt(normal_x**2 + normal_y**2)
# normal_x /= length
# normal_y /= length

# # Extend the line by 100 pixels in both directions (First and Second Centroids)
# x1_ext = int(centroid1[0] + 250 * normal_x)
# y1_ext = int(centroid1[1] + 250 * normal_y)
# x2_ext = int(centroid2[0] + 250 * normal_x)
# y2_ext = int(centroid2[1] + 250 * normal_y)

# # Draw the extended line between the first and second centroids
# cv2.line(imageFrame, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 255, 0), 2)

# # Compute the vector from centroid1 to centroid3
# dx3 = centroid3[0] - centroid1[0]
# dy3 = centroid3[1] - centroid1[1]
# print(dy3/dx3)

# # Compute the normal (perpendicular) vector for the first and third centroids
# normal_x3 = -dy3
# normal_y3 = dx3

# # Normalize the normal vector
# length3 = np.sqrt(normal_x3**2 + normal_y3**2)
# normal_x3 /= length3
# normal_y3 /= length3

# # Extend the line by 100 pixels in both directions (First and Third Centroids)
# x1_ext3 = int(centroid1[0] + 250 * normal_x3)
# y1_ext3 = int(centroid1[1] + 250 * normal_y3)
# x3_ext = int(centroid3[0] + 250 * normal_x3)
# y3_ext = int(centroid3[1] + 250 * normal_y3)

# # Draw the extended line between the first and third centroids
# cv2.line(imageFrame, (x1_ext3, y1_ext3), (x3_ext, y3_ext), (255, 0, 0), 2)

# # Compute the Euclidean distance between centroid1 and centroid3
# distance = int(
#     np.sqrt((centroid3[0] - centroid1[0])**2 + (centroid3[1] - centroid1[1])**2))

# # Get the vertical center of the image
# center_x = imageFrame.shape[1] // 2  # Middle of the width

# # Determine the start and end y-coordinates for the vertical line
# start_y = (imageFrame.shape[0] - distance) // 2
# end_y = start_y + distance

# # Draw a vertical line at the center of the image with the same length as centroid1 to centroid3
# cv2.line(imageFrame, (center_x, start_y), (center_x, end_y),
#          (255, 0, 255), 2)  # Magenta color


# # Show the final image with the two extended lines
# cv2.imshow("Detected Yellow Colour with Extended Lines", imageFrame)

# # Wait for a key press and close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# color_ranges = {
#     "Yellow": ((25, 50, 50), (40, 255, 255), (0, 255, 255)),
# }

# # Load the image
# image_path = "/home/antonio/Pictures/Screenshots/curve_far.png"
# imageFrame = cv2.imread(image_path)

# if imageFrame is None:
#     print(f"Error: Could not load image from {image_path}")
#     exit()

# # Get screen size
# screen_width, screen_height = pyautogui.size()

# # Resize the image to 20% of the screen size
# new_width = int(screen_width * 0.2)
# new_height = int(screen_height * 0.2)
# imageFrame = cv2.resize(imageFrame, (new_width, new_height))

# # Convert to HSV color space
# hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# # Detect yellow color
# yellow_lower = np.array(color_ranges["Yellow"][0], np.uint8)
# yellow_upper = np.array(color_ranges["Yellow"][1], np.uint8)
# yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

# # Morphological processing
# yellow_mask = cv2.dilate(yellow_mask, np.ones((5, 5), "uint8"))


# def find_centroids(mask):
#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     centroids = []
#     for contour in contours:
#         if cv2.contourArea(contour) > 300:
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 centroids.append((cX, cY))
#     return centroids


# # Get centroids and sort them
# centroids = find_centroids(yellow_mask)
# if len(centroids) < 3:
#     print("Not enough centroids detected!")
#     exit()

# centroids.sort(key=lambda c: np.sqrt(
#     c[0]**2 + (imageFrame.shape[0] - c[1])**2))
# centroid1, centroid2, centroid3 = centroids[:3]

# # Bottom center of the image
# bottom_center = (imageFrame.shape[1] // 2, imageFrame.shape[0])

# # Compute direction vectors
# vec12 = np.array([centroid2[0] - centroid1[0], centroid2[1] - centroid1[1]])
# vec13 = np.array([centroid3[0] - centroid1[0], centroid3[1] - centroid1[1]])

# # Normalize the vectors
# vec12 = vec12 / np.linalg.norm(vec12)
# vec13 = vec13 / np.linalg.norm(vec13)

# # Scale the vectors (change 250 to adjust length)
# line_length = 250
# end12 = (bottom_center[0] + int(line_length * vec12[0]),
#          bottom_center[1] + int(line_length * vec12[1]))
# end13 = (bottom_center[0] + int(line_length * vec13[0]),
#          bottom_center[1] + int(line_length * vec13[1]))

# # Compute the endpoint for the pink vertical line
# vertical_length = 250  # Adjust as needed
# end_vertical = (bottom_center[0], bottom_center[1] - vertical_length)

# # Draw the lines from bottom center
# cv2.line(imageFrame, bottom_center, end12, (0, 255, 0), 2)  # Green line
# cv2.line(imageFrame, bottom_center, end13, (255, 0, 0), 2)  # Blue line
# cv2.line(imageFrame, bottom_center, end_vertical,
#          (255, 0, 255), 2)  # Pink vertical line

# # Show the final image
# cv2.imshow("Yellow Color Detection with Lines", imageFrame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
