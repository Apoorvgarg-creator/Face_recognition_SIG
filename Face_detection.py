import cv2
import numpy as np

name = input("Enter Name: ")

face_data = []
# .npy

cap = cv2.VideoCapture(0)


# Convolutional neural networks-> Resnet50  and Pretrained Model  --> haarcasscade_frontalface

face_cascade = cv2.CascadeClassifier("/Users/apoorvgarg/PycharmProjects/SIG11Sept/Face_recognition/haarcascade_frontalface_alt.xml")

while True: # 4 spaces or 1 tab
    ret, frame = cap.read() # ret -> status, frame -> 2d image

    if ret == False:
        continue

    # ROI -> Region of interest
    # here -> face
    faces = face_cascade.detectMultiScale(frame, 1.3, 5) # Frame , 1.3 scaling factor, 5 -> neighbour
    # Image  ->  3 values -> x,y,z
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    faces = faces[:1]

    for face in faces:
        x,y,w,h = face # tuple unpacking
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        # extracting the face
        face_ = frame[y:y+h,x:x+w]
        # [] -> list
        # () -> tuple
        # {} -> dict # HashMap # key-value pair

        face_ = cv2.resize(face_,(32,32))
        face_data.append(face_)


    key = cv2.waitKey(1) # 1ms delay
    if key & 0xFF == ord('q'):  # Ascii value of q -> 0xFF
        break



print(len(face_data))

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(("dataset/"+name),face_data)
cap.release()
cv2.destroyAllWindows()






