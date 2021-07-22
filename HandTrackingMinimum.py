import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()

    # Mediapipe provides drawing_utils for Drawing landmarks
mpDraw = mp.solutions.drawing_utils         # mpDraw is the object for drawing the landmarks

# For Frame Rate Calculation
prev_Time = 0
current_Time = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)

    ''' Calculation the hand number and drawin the connection between those lines'''
    if result.multi_hand_landmarks:                         # Detecting multiple hands on the vedio
        for handLmarks in result.multi_hand_landmarks:      # For each hand land marks is calculated it store in 'handLmarks'
            # Enumarate all the land marks and print them
            for id, lm in enumerate(handLmarks.landmark):   # id = serial number, lm = x,y,z coordinate value
                # print(id,lm)                                # 20 x: 0.1742537021636963 ,y: 0.1360754668712616, z: -0.04216291010379791
                h, w, c = img.shape         # It gives the height and the width and the chennal of the image capture by camera
                cx, cy = int(lm.x*w), int(lm.y*h)           # Getting the x and y coordinate of the hand landmark
                print(id, cx, cy)               # Printing the x and y cordinate of each landmark point
                if id == 4:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)   # Draw a circle at the 0th landmark of the hand


            mpDraw.draw_landmarks(img,handLmarks, mpHand.HAND_CONNECTIONS)  # Draw the land marks and also connect all those lines


    ''' Calculating the frame time and print it on the frame '''
    current_Time = time.time()
    frame_per_seconod = 1/(current_Time - prev_Time)
    prev_Time = current_Time
        # Printing the Frame per second
    cv2.putText(img, str(int(frame_per_seconod)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
