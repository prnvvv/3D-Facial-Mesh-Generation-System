import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)



previousTime = 0
currentTime = 0

while True:

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.imshow()

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()