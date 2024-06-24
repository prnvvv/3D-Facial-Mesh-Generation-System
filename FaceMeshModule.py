import cv2  # OpenCV library for computer vision tasks
import mediapipe as mp  # MediaPipe library for face mesh detection
import time  # Time library to calculate frame rate

# FaceMeshDetector class for detecting face mesh and finding face landmarks
class FaceMeshDetector:
    def __init__(self, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize video capture object to read from webcam
        self.cap = cv2.VideoCapture(0)

        # Check if the webcam is opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Initialize MediaPipe face mesh solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Initialize MediaPipe drawing utility
        self.mpDraw = mp.solutions.drawing_utils
        # Define drawing specifications for landmarks
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    # Method to detect face mesh and draw landmarks
    def detectFaceMesh(self):
        previousTime = 0

        while True:
            # Read frame from webcam
            success, vidObject = self.cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            # Convert the frame to RGB as MediaPipe uses RGB format
            imgRGB = cv2.cvtColor(vidObject, cv2.COLOR_BGR2RGB)
            # Process the frame to detect face mesh
            results = self.face_mesh.process(imgRGB)

            # Get the face landmarks
            detections = results.multi_face_landmarks

            if detections:
                # Draw face landmarks if faces are detected
                for detection in detections:
                    self.mpDraw.draw_landmarks(
                        vidObject, detection, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec
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
        self.cap.release()
        cv2.destroyAllWindows()

# Entry point of the script
if __name__ == "__main__":
    # Initialize FaceMeshDetector object
    face_mesh_detector = FaceMeshDetector(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Start detecting face mesh
    face_mesh_detector.detectFaceMesh()
