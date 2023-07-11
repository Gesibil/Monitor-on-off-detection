import cv2
import numpy as np

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(2)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit(1)
# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    # Perform dilation on the thresholded image
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(threshold, rectKernel)
    edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
    # Detect all contours in Canny-edged image
    _,contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    monitor_contour= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    receipt_contour=get_receipt_contour(monitor_contour)
    image= cv2.drawContours(frame.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    cv2.imshow('Black Region Detection', image)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
