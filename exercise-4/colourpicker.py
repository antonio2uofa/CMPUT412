import cv2

# Callback function for mouse click events


def click_event(event, x, y, flags, param):
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the pixel at the clicked location in HSV
        hsv_color = hsv_image[y, x]
        # Print the color (HSV format)
        print(
            f"Clicked pixel at ({x}, {y}) with color: HSV({hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]})")

        # Display the clicked color in the window
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"HSV: {hsv_color[0]}, {hsv_color[1]}, {hsv_color[2]}"
        cv2.putText(image, text, (x + 10, y + 10),
                    font, 0.5, (255, 255, 255), 2)

        # Show the updated image
        cv2.imshow("Image", image)


# Load the image
image_path = "/home/antonio/Pictures/Screenshots/red_dark_2.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Resize image to fit the screen (adjust this percentage as needed)
scale_percent = 20  # Adjust percentage to fit your screen
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize for easier viewing
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert the resized image to HSV
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Show the image and set up the mouse callback
cv2.imshow("Image", resized_image)
cv2.setMouseCallback("Image", click_event)

# Wait indefinitely for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
