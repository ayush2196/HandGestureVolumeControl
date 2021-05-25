import cv2
import time
import numpy as np
import handTracyModule as ht
import math
import osascript

wCam, hCam = 640, 480  # To set the height and width of the Video Frame

cap = cv2.VideoCapture(0)  # Capture video from integrated webcam
cap.set(3, wCam)
cap.set(4, hCam)
previousTime = 0

detector = ht.handDetection(detectionConfidence=0.7)

# result = osascript.osascript('get volume settings')

minVol = 0
maxVol = 100
vol = 0
barVol = 350
volPercentage = 0

# osascript.osascript("set volume output volume 100")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        # print(landmarkList[4], landmarkList[8])
        x1, y1 = landmarkList[4][1], landmarkList[4][2]  # Thumb
        x2, y2 = landmarkList[8][1], landmarkList[8][2]  # Index Finger
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # To find out the center between thumb and index finger

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  # Finding length of the line between thumb and index finger
        # print(length)

        vol = np.interp(length, [20, 170], [minVol, maxVol])
        # Converting the volume range to the range of distance between thumb and index finger
        barVol = np.interp(length, [20, 170], [350, 100])
        volPercentage = np.interp(length, [20, 170], [0, 100])
        print(int(length), vol)
        vol = "set volume output volume " + str(vol)
        osascript.osascript(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 100), (85, 350), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(barVol)), (85, 350), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPercentage)} %', (40, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
