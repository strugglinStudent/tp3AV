import cv2
import numpy as np
from tracker import *
# Create tracker object
tracker = EuclideanDistTracker()

# Open the video containing the binary mask
mask_video = cv2.VideoCapture('video.mp4')  # Replace with your mask video file
n = 60
# Open the video you want to label based on the mask
input_video = cv2.VideoCapture('vid.mp4')  # Replace with your input video file

# Get video properties
fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
for _ in range(n):
    ret_input, _ = input_video.read()
while True:
    ret_mask, frame_mask = mask_video.read()
    ret_input, frame_input = input_video.read()
    detections = []
    if not ret_mask or not ret_input:
        break
    
    # Convert the mask frame to grayscale
    mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

    # Threshold the mask frame to get a binary mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the contours on the input frame
    frame_with_rectangles = frame_input.copy()
    for i, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter out contours with an area smaller than 500
        if area < 11000 or area>15000:
            continue
        #print(area,"\n")    
        x, y, w, h = cv2.boundingRect(contour)
        detections.append((x, y, x + w, y + h))

        # Label each object with a unique ID and its contour area


        boxes_ids = tracker.update(detections)
    prev_center = (int((x + x + w) / 2), int((y + y + h) / 2))
    # Draw the updated bounding boxes on the frame
    for box_id in boxes_ids:
        x, y, w, h, obj_id = [int(c) for c in box_id]

        # Draw rectangle around the object
        cv2.rectangle(frame_with_rectangles, (x, y), (w, h), (0, 255, 0), 2)

        # Show object ID
        cv2.putText(frame_with_rectangles, f"ID: {obj_id}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        current_center = (tracker.center_points[obj_id])
        print(tracker.center_points[obj_id])
        cv2.line(frame_with_rectangles, prev_center, current_center, (0, 0, 255), 1)
        prev_center = current_center


    # Display the frame with rectangles and labels
    cv2.imshow("Objects in Video", frame_with_rectangles)

    # Write the frame to the output video
    output_video.write(frame_with_rectangles)

    # Break the loop if the 'ESC' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
mask_video.release()
input_video.release()
output_video.release()
cv2.destroyAllWindows()
