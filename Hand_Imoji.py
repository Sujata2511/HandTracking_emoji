"""
Here we Show the Gesture Emoji according to our hand gesture:---
"""


import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 600, 900           # wCam = Weidth of the Camera, hCam = Height of the Camera

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, wCam)

'''     --: Loading all the image from a directory to a list in our program :--         '''
folderPath = "Hand_Images"
myList = os.listdir(folderPath)     # With the help of the os module we make the list all the image name in the myList
print(myList)                       # e.g: ['a.png', 'B.png', 'c.png', 'D.png', 'E.png']
overLayList = []
for imgPath in myList:                                  # Iterate the image name one by one
    image = cv2.imread(f"{folderPath}/{imgPath}")       # Making the full URL of the image
    # print(f"{folderPath}/{imgPath}")                  # Printing all the image URL of all the image
    overLayList.append(image)                           # List all the image in the overLayList list
print(len(overLayList))                                 # print the total number of image
pTime = 0

detector = htm.handDetector(detectionCon=0.75)           # Detection Confidence = 75%
tipIds = [4, 8, 12, 16, 20]                              # Tip Landmark of all the 5 fingers

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    '''     --: Checking if the finger is open or not :--       '''
    fingersTip = []
    if len(lmList) !=0:
        # Thumb checking
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # Comparing the value according to the x axis value
            fingersTip.append(tipIds[0])

        # Rest of 4 fingers
        for id in range(1,5):                               # Checking the tip of 5 finger
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:     # Checking the if the finger is open or not
                fingersTip.append(tipIds[id])                # Then put 1 in the list

        if len(fingersTip) == 0:
            fingersTip.append(0)
        print(fingersTip)                                    # Print the finger array

    Finger = 10
    '''     Find Which finger is Up  '''
    if fingersTip == [0]:                   # Raised
        Finger = 1
    elif fingersTip == [4]:                 # Thumbs Up
        Finger = 2
    elif fingersTip == [8]:                 # Index Finger
        Finger = 3
    elif fingersTip == [4,8]:               # Backhand Index
        Finger = 4
    elif fingersTip == [4,20]:              # Call Me
        Finger = 5
    elif fingersTip == [8,20]:              # Horns Hand
        Finger = 6
    elif fingersTip == [8,12]:              # Victory
        Finger = 7
    elif fingersTip == [4,8,20]:            # Love
        Finger = 8
    elif fingersTip == [4,8,12,16,20]:      # Hand Raised
        Finger = 9
    else: Finger = 10

    '''     --: Display the overlap image :--       '''
    height, width, channel = overLayList[Finger-1].shape   # .shape gives us the hight, width, channel of the image
    img[0:height, 0:width] = overLayList[Finger-1]         # Here we slicing the image the  imge[height,width]

    '''     --: Puting the frame/second on the vedio :--    '''
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,f'FPS: {int(fps)}',(730,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,34),2)

    cv2.imshow("image", img)
    cv2.waitKey(1)

