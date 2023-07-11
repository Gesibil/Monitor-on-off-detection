import cv2
import numpy as np

# Define the dimensions of the LED rectangle
led_width = 0.12   # in cm
led_height = 0.5   # in cm

# Load the image
image = cv2.imread('monitor.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Failed to load image")
    exit(1)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the image
_, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Apply color filtering to extract white regions
lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(image, lower_white, upper_white)

# Bitwise AND the thresholded image and the color mask
filtered = cv2.bitwise_and(threshold, threshold, mask=mask)

# Find contours in the thresholded image
contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cm_to_pixel = 0.0264583333
# Iterate over the contours and filter based on area and aspect ratio
i=0
# Iterate over the contours and filter based on area and shape
for contour in contours:
    
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    i=i+1
    if len(approx) == 4:
        # Draw a rectangle around the detected object
        x, y, w, h = cv2.boundingRect(contour)
        if w==6 and h==17:
        
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print( "led was detected",w, h)
            cv2.drawContours(image, [np.array([(x-10, y-10), (x + w+10, y-10), (x + w+10, y + h+20), (x-10, y + h+20)])], -1,(0, 255, 0) , 4)
            cv2.putText(image, 'LED Detected', (x-100, y + h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



image=cv2.resize(gray, (820, 610))

# Display the result
cv2.imshow('LED Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()







