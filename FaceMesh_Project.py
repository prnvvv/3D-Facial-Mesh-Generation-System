import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

previousTime = 0
currentTime = 0

while True:
    success, vidObject = capture.read()

    imgRGB =    cv2.cvtColor(vidObject, cv2.COLOR_BGR2RGB)

    results = FaceMesh.process(imgRGB)

    detections = results.multi_face_landmarks
    
    if detections:
        for detection in detections:
            mpDraw.draw_landmarks(vidObject, detection, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
            for landmark in detection.landmark:
                print(landmark)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    
    cv2.putText(vidObject, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", vidObject)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()