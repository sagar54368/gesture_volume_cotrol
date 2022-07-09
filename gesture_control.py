import cv2 
import mediapipe as mp
import numpy as np
import time
import Hand_tracking_module as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#########################
wCam, hcam = 840, 680
#########################


cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hcam)
ptime=0

detector = ht.handdetector()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
print(volRange)

minVol = volRange[0]
maxVol = volRange[1]
volbar = 400
volper = 0
while True:
    success, img = cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList)!=0:
        # print value of list at any index(landmark)
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1] , lmList[4][2]
        x2, y2 = lmList[8][1] , lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles around 2 pts and connect with line
        cv2.circle(img, (x1,y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 50, 200), 3)
        cv2.circle(img, (cx,cy), 10, (0, 255, 255), cv2.FILLED)

        #calculate length 
        length = math.hypot(x2-x1, y2 - y1)
        #print(length)


        #Hand range 20 to 200
        # volume range -65 to 0
        vol = np.interp(length, [20, 200], [minVol, maxVol])
        volbar = np.interp(length, [20, 200], [400, 150])
        volper = np.interp(length, [20, 200], [0, 100])


        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)


        if length<30:
            cv2.circle(img, (cx,cy), 10, (0, 40, 100), cv2.FILLED)

    # draw rectangle to view vol bar
    cv2.rectangle(img, (50, 150), (85, 400),(0, 0, 0), 4)
    cv2.rectangle(img, (50, int(volbar)), (85, 400),(0, 0, 0), cv2.FILLED)
    cv2.putText(img,f'{int(volper)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255), 3)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2,
                    (0, 0, 255), 1)

    cv2.imshow("Image",img)
    cv2.waitKey(1)