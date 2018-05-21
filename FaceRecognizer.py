"""
Email: siddharthchandragzb@gmail.com
"""

import cv2
import os
import numpy as np


class recognizer(object):

    def __init__(self,X,y):
        self.X = X
        self.y = y

    def recognize(self, model):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

        while True:

            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                roi = gray[y-20:y+h+20, x-20:x+w+20]
                
                try:
                    roi = cv2.resize(roi, (200,200), interpolation = INTER_LINEAR)
                    params = model.predict(roi)

                    print("Label: ", params[0], " Conf scr: ", params[1])
                    if params[0] == 0:
                        cv2.putText(frame, "Sid", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "UnKnown", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2)

                except:
                    continue

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

def read_database(path = "./database"):

    images = os.listdir(path)
    img, idx = [], []
    for image in images:
        im = cv2.imread(path+image, cv2.IMREAD_GRAYSCALE)
        img.append(im)
        idx.append(0)

    return img, idx

if __name__ == "__main__":

    X,y = read_database()

    Obj = recognizer()
    
    Eigenmodel = cv2.face.createEigenFaceRecognizer()
    Eigenmodel.train(np.asarray(X), np.asarray(y))
    Obj.recognize(Eigenmodel)
    
    Fishermodel = cv2.face.createFisherFaceRecognizer()
    Fishermodel.train()
    Obj.recognize(Fishermodel)
    
    LBPHmodel = cv2.face.createLBPHFaceRecognizer()
    LBPHmodel.train(np.asarray(X), np.asarray(y))
    Obj.recognize(LBPHmodel)
    
