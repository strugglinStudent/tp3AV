import pydot
import cv2
import cv2
import datetime
# Open the video
video = cv2.VideoCapture('./vid.mp4')  # Replace with your video file
n=60
# Initialize a list to store all frames
all_frames = []
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc(*'H264')
video_output = cv2.VideoWriter(('videoseg.mp4'), video_codec, fps, (frame_width, frame_height))
print(video_codec, fps, frame_width, frame_height)
while True:
    ret, frame = video.read()

    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Append the current frame to the list of all frames
    all_frames.append(gray_frame)

# Calculate differences between frame[x-n] and frame[x+n] for each frame
for i in range(n, len(all_frames) - n):
    frame_x_minus_n = all_frames[i - n]
    frame_x_plus_n = all_frames[i + n]

    # Calculate the difference between the two frames
    frame_diff_mn = cv2.absdiff(frame_x_minus_n, all_frames[i])
    _, binary_diff_mn = cv2.threshold(frame_diff_mn, 60, 255, cv2.THRESH_BINARY)

    frame_diff_pn = cv2.absdiff(frame_x_plus_n, all_frames[i])
    _, binary_diff_pn = cv2.threshold(frame_diff_pn, 60, 255, cv2.THRESH_BINARY)

    frame_diff = binary_diff_mn & binary_diff_pn
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust the kernel size as needed
    frame_diff_opened = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, kernel)
     # Find contours in the cleaned binary difference frame
    # Apply morphological opening to eliminate small zones
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Adjust the kernel size for opening
    frame_diff_cleaned = cv2.morphologyEx(frame_diff_opened, cv2.MORPH_CLOSE, kernel_open)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))  # Adjust the kernel size for opening
    frame_diff_cleaned = cv2.morphologyEx(frame_diff_cleaned, cv2.MORPH_OPEN, kernel_open)
    cv2.imshow("frames tn,t+n",frame_diff_cleaned )
    video_output.write(frame_diff_cleaned)
    
    if cv2.waitKey(30) & 0xFF == 27:  # Wait for 30 ms, if 'ESC' key is pressed, exit the loop
        break
# Release the video and close windows
print (i,"\n")
video.release()
video_output.release()
cv2.destroyAllWindows()
