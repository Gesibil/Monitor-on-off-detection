import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('monitor.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Failed to load image")
    exit(1)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to get edge contours
edges = cv2.Canny(gray, 100, 200)

# Apply color filtering to detect white regions
lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
white_mask = cv2.inRange(image, lower_white, upper_white)

# Find contours in the white mask
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

# Resize the images
original_image = cv2.resize(image, (820, 610))
edge_contour_image = cv2.resize(edges, (820, 610))
white_mask_image = cv2.resize(white_mask, (820, 610))

# Create a figure and plot the images
fig, axes = plt.subplots(1, 3)
axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[1].imshow(edge_contour_image, cmap='gray')
axes[1].set_title('Edge Contour Image')
axes[2].imshow(white_mask_image, cmap='gray')
axes[2].set_title('White Mask Image')

# Remove the axis labels
for ax in axes:
    ax.axis('off')

# Adjust the layout and display the figure
plt.tight_layout()
plt.show()
