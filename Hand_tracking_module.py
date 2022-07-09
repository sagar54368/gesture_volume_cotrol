import cv2 
import mediapipe as mp
import time


class handdetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #using Mediapipe Solutions module
        self.mpHands = mp.solutions.hands    
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        #for drawing ponts and connectins btw points on  hands
        self.mpdraw = mp.solutions.drawing_utils


    #detection part
    def findHands(self, img, draw=True):
        #MediaPipe only uses RGB images but Opencv here gives us BGR images so we convert it to RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.output = self.hands.process(imgRGB)

        #now we check if the output has multiple hands and try to get the landmarks of those hands
        if self.output.multi_hand_landmarks:
            for handsLM in self.output.multi_hand_landmarks:
               #for Drwaing joints and lines connecting those joints(21 Points of interest)
                if draw:
                    self.mpdraw.draw_landmarks(img,handsLM,self.mpHands.HAND_CONNECTIONS)
        return img


    #find list(position) of landmarks
    def findPosition(self, img, handNo=0, draw=True,id_lm=0):
        #id_lm for particular point highlight
        id=0
        lmList = []

        if self.output.multi_hand_landmarks:
            #which hand are we taking about(for 1 particular hand)
            myHand= self.output.multi_hand_landmarks[handNo]
            #Information for each hand in video using Landmark(gives index and X&Y coordinates)
            for id ,lm in enumerate(myHand.landmark):
                #The model has 21 handmarks and this each landmarks id number along with X,y,z coordinates
                h, w, c = img.shape
                #The model returns decimal coordinates which are a ratio of height and width so we multiply with H and W to get Pixel VAlues and precise Location
                cx, cy = int(lm.x*w),int(lm.y*h) #Centre X and Y for each 21 Landmarks
                #print("ID :",id,"Cx:",cx,"CY:",cy)
                lmList.append([id, cx, cy])
                if draw:
                    if id == id_lm:
                        #By using this we can highlight any particle Landmk and use it 
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        return lmList

def main():
    ptime = 0;
    ctime = 0;

    #For video capturing using webcam
    cam = cv2.VideoCapture(0)
    detector = handdetector()

    while True :
        success,img = cam.read()
        img = detector.findHands(img)
        lmList=detector.findPosition(img,id_lm=4)
        if len(lmList)!=0:
            #print value of list at any index(landmark)
            print(lmList[1])

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        #fps = str(int(fps)) 
        #cv2.putText(img,str(fps),(10,100),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255),1)
        cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3,
                    (255, 0, 255), 3)
    
        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
