"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import os
import numpy as np
import re

class recognizer(object):

    def __init__(self):
        pass

    def recognize(self, model):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
        name = ["Sid", "Abhi"]

        while True:

            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                roi = gray[y-20:y+h+20, x-20:x+w+20]

                try:
                    roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_LINEAR)
                    params = model.predict(roi)
                    cv2.putText(frame, name[params[0]], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

                except:
                    continue
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

def read_database(path = "./database/"):

    images = os.listdir(path)
    img, idx = [], []
    for image in images:
        im = cv2.imread(path+image, cv2.IMREAD_GRAYSCALE)
        img.append(im)
        #print(re.search(r"([0-9]{1,3})", path+image).group())

        if int(re.search(r"([0-9]{1,3})", path+image).group())>150:
            idx.append(1)
        else:
            idx.append(0)

        print(idx)

    return img, idx

if __name__ == "__main__":

    X,y = read_database()

    Obj = recognizer()
    
    Eigenmodel = cv2.face.EigenFaceRecognizer_create()
    Eigenmodel.train(np.asarray(X), np.asarray(y))
    Obj.recognize(Eigenmodel)
    
    Fishermodel = cv2.face.FisherFaceRecognizer_create()
    Fishermodel.train(np.asarray(X), np.asarray(y))
    Obj.recognize(Fishermodel)
    
    LBPHmodel = cv2.face.LBPHFaceRecognizer_create()
    LBPHmodel.train(np.asarray(X), np.asarray(y))
    Obj.recognize(LBPHmodel)
    
