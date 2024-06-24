import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for face mesh detection
import time  # Time library to calculate frame rate

# Initialize video capture object to read from webcam
capture = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not capture.isOpened():
    print("Webcam could not be opened.")
    exit()

# Initialize MediaPipe face mesh solution
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# Initialize MediaPipe drawing utility
mpDraw = mp.solutions.drawing_utils
# Define drawing specifications for landmarks
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# Variables to calculate frames per second (FPS)
previousTime = 0
currentTime = 0

while True:
    # Read frame from webcam
    success, vidObject = capture.read()

    if not success:
        break
    
    # Convert the frame to RGB as MediaPipe uses RGB format
    imgRGB = cv2.cvtColor(vidObject, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face mesh
    results = FaceMesh.process(imgRGB)

    # Get the face landmarks
    detections = results.multi_face_landmarks
    
    if detections:
        # Draw face landmarks if faces are detected
        for detection in detections:
            mpDraw.draw_landmarks(
                vidObject, detection, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec
            )
            # Print each landmark's position
            for landmark in detection.landmark:
                print(landmark)

    # Calculate FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    
    # Display FPS on the frame
    cv2.putText(vidObject, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with face landmarks
    cv2.imshow("Video", vidObject)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
