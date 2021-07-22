import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode = False,maxHands = 2,detectionCon = 0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:                                                       #detecting multiple hands on the vedio
            for handLmarks in self.result.multi_hand_landmarks:                                    # For each hand land marks is calculated it store in 'handLmarks'
                if draw:
                    self.mpDraw.draw_landmarks(img,handLmarks, self.mpHand.HAND_CONNECTIONS)  # Draw the land marks and also connect all those lines
        return img

    def findPosition(self, img, handNo=0, draw = True):

        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    prev_Time = 0
    current_Time = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) !=0:
            print(lmList[0])

        current_Time = time.time()
        frame_per_seconod = 1 / (current_Time - prev_Time)
        prev_Time = current_Time
        cv2.putText(img, str(int(frame_per_seconod)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
