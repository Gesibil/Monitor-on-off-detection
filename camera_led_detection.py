import cv2
import numpy as np

# Define the dimensions of the LED rectangle
led_width = 0.12   # in cm
led_height = 0.5   # in cm

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture('monitor.jpg')

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

# Loop to continuously process video frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Failed to read frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the frame
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Apply color filtering to extract white regions
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame, lower_white, upper_white)

    # Bitwise AND the thresholded frame and the color mask
    filtered = cv2.bitwise_and(threshold, threshold, mask=mask)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cm_to_pixel = 0.0264583333

    # Iterate over the contours and filter based on area and shape
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) == 4:
            # Draw a rectangle around the detected object
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - (led_width / cm_to_pixel)) < 5 and abs(h - (led_height / cm_to_pixel)) < 5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(frame, [np.array([(x-10, y-10), (x + w+10, y-10), (x + w+10, y + h+20), (x-10, y + h+20)])], -1, (0, 255, 0), 4)
                cv2.putText(frame, 'LED Detected', (x-100, y + h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('LED Detection', frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
