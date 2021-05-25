import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLandmarks in result.multi_hand_landmarks:
            for id, landmarks in enumerate(handLandmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmarks.x*w), int(landmarks.y*h)
                #print(id, cx, cy)

                
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
