import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # mphands means multiple hands
mpDraw = mp.solutions.drawing_utils # method to draw the points and line between them

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) # means we are giving the image to be processed and then we are extracting the results
    # print(results.multi_hand_landmarks)
    # multi_hand_landmark is used for getting the landmarks of the palm

    if results.multi_hand_landmarks:  #
        for handLms in results.multi_hand_landmarks: # handLms means for each hand landmark
            for id, lm in enumerate(handLms.landmark): # get the id no
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
             # we are giving the original image Not RGB
             # mphands.Hand_Connection mainly draw the connection btwn the points

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)