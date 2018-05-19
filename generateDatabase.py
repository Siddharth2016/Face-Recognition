"""
Email: siddharthchandragzb@gmail.com
"""

import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

count = 40

while True and count<51:

    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)

    for (x,y,w,h) in faces:

        roi = grayframe[y-20:y+h+20, x-20:x+h+20]
        roi = cv2.resize(roi, (200,200))

        cv2.imwrite("./database/{}.pgm".format(count), roi)

    count += 1
    
    cv2.imshow("Your Face", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()
