# Install packages
import cv2
import time

# Create first frame video
frame_1 = None

# Set up the video Capture
video = cv2.VideoCapture(0)

# Create the loop for checking the capture
while True:

    # Get the frame from the video
    check, frame = video.read()

    # Grayscale the image and blur it to help remove noise
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    gray = cv2.GaussianBlur(
        gray,
        (23, 23),
        0
    )

    # Create a conditional for gettig the first frame
    if frame_1 is None:
        frame_1 = gray
        # Send the code back to the start of the loop if it is the first time
        continue

    # Compare the first frame with the current frame
    frame_diff = cv2.absdiff(
        frame_1,
        gray
    )

    # Add a threshold to the values
    threshold_frame_diff = cv2.threshold(
        frame_diff,
        30,
        255,
        cv2.THRESH_BINARY
    )[1]

    # Smooth out the threshold frame
    threshold_frame_diff = cv2.dilate(
        threshold_frame_diff,
        None,
        iterations=3
    )

    # Find the contours
    (contours, p) = cv2.findContours(
        threshold_frame_diff.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Make sure we only use large contours
    for con in contours:
        if cv2.contourArea(con) < 1000:
            continue

        # Plot the contour
        (x, y, w, h) = cv2.boundingRect(con)
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 255),
            2
        )

    # Display the images
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference", frame_diff)
    cv2.imshow("Threshold image", threshold_frame_diff)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    print(gray)

    # Create an escape method
    if key == ord("q"):
        break

# Release the video
video.release()

# Get rid of the windows opened by the script
cv2.destroyAllWindows
