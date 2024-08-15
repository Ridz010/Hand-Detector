from handDetection import HandDetection

import cv2

handDetection = HandDetection(min_detection_confidence=0.5, min_tracking_confidence=0.5)

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    status, frame = webcam.read()
    
    if not status or frame is None:
        print("Error: Could not read frame")
        break
    
    frame = cv2.flip(frame, 1)
    
    cv2.imshow("Hand Landmark", frame)

    handLandMarks = HandDetection.findhandLandMark(image=frame, draw=True)
    
    if cv2.waitKey(1) == ord('a'):
        break

webcam.release()
cv2.destroyAllWindows()