import cv2
import numpy as np
import os
from PIL import Image


class Fc_Rec(object):
    
    
    def __init__(self):
        self.faceDetector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
        self.cam=cv2.VideoCapture(1)
        self.recognizer= cv2.face.LBPHFaceRecognizer_create()
        self.path='dataset'\


    def register(self,id):
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        sampleNum=0
        while True:
            ret,img=self.cam.read()
            face,Coords=self._get_face_image_cv(img,self.faceDetector)
            if face is not None:
                (x,y,w,h)=Coords
                sampleNum=sampleNum+1
                writepath="dataset/user."+str(id)+"."+str(sampleNum)+".jpg"
                im_pil = Image.fromarray(face)
                im_pil.save(writepath)
                #cv2.imwrite("dataset/user."+str(id)+"."+str(sampleNum)+".jpg",face)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
                cv2.waitKey(100)
            cv2.imshow("Face",img)
            cv2.waitKey(1)
            if(sampleNum>20):
                break
        cv2.destroyAllWindows()


    def training(self):
        Ids, faces=self._getImageWithID(self.path)
        self.recognizer.train(faces,Ids)
        self.recognizer.save('recognizer/trainingData.yml')



    def detect(self):
        self.recognizer.read("recognizer/trainingData.yml")
        id=0
        ids=[]
        font=cv2.FONT_HERSHEY_COMPLEX
        sampleNum=0

        while True:
            ret,img=self.cam.read()
            face,Coords=self._get_face_image_cv(img,self.faceDetector)
            if face is not None:
                sampleNum=sampleNum+1
                (x,y,w,h)=Coords
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
                id,conf=self.recognizer.predict(face)
                ids.append(id)
                cv2.putText(img,str(id),(x,y+h),font,1,(255,0,0),2)
            cv2.imshow("Face",img)
            cv2.waitKey(1)
            if(sampleNum>20):
                break

        return max(set(ids), key = ids.count)
        ids=[]
        cv2.destroyAllWindows()


    def Test_Camera(self):
        while True:
            ret,img=self.cam.read()
            face,Index=self._get_face_image_cv(img,self.faceDetector)
            if face is not None:
                (x,y,w,h)=Index
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
            cv2.imshow("Face",img)
            if(cv2.waitKey(1)==ord('q')):
                break
        cv2.destroyAllWindows()    

    @staticmethod
    def _getImageWithID(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        faces=[]
        IDs=[]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L')
            faceNp=np.array(faceImg)
            ID=int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            #cv2.imshow("training",faceNp)
            #cv2.waitKey(10)
        return np.array(IDs), faces

    @staticmethod
    def _get_face_image_cv(camera_image,face_detector):

        img_gray = cv2.cvtColor(camera_image, cv2.COLOR_BGR2GRAY, 1)
        faces = face_detector.detectMultiScale(img_gray,1.3,5)

        # Crop the first face found
        if len(faces):
            x, y, w, h = faces[0].tolist()
            face_image = img_gray[y:y + h, x:x + w]
            return (face_image, (x, y, w, h))

        return (None, None)



    def __del__(self):
        self.cam.release()
