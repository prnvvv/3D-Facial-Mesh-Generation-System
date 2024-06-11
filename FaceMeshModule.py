import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def detectFaceMesh(self):
        previousTime = 0

        while True:
            success, vidObject = self.cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            imgRGB = cv2.cvtColor(vidObject, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(imgRGB)

            detections = results.multi_face_landmarks

            if detections:
                for detection in detections:
                    self.mpDraw.draw_landmarks(
                        vidObject, detection, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec
                    )
                    for landmark in detection.landmark:
                        print(landmark)

            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            cv2.putText(vidObject, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video", vidObject)

            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_mesh_detector = FaceMeshDetector(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh_detector.detectFaceMesh()
