import cv2
import numpy as np
import face_recognition
from keras.models import load_model
from yolov8 import YOLOv8
import os
from datetime import datetime
import imutils as im

# Initialize YOLOv8 object detector
model_path = "/Users/apple/Desktop/final_year_project/AI-Enabled-Student-Tracking-System/models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

model=load_model('/Users/apple/Desktop/final_year_project/AI-Enabled-Student-Tracking-System/saved_model 20-01-22.h5')

def load_known_images(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode = face_encodings[0]
            encodeList.append(encode)
        else:
            encodeList.append(None)
    return encodeList

def mark_attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def perform_face_recognition(frame, encodeListKnown, classNames):
    resized_frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)  # Resize frame for faster processing
    
    imgS = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2  # Scale up bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    return frame



def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

# Load known images and their encodings
known_images, class_names = load_known_images('ImagesAttendance')
encode_list_known = find_encodings(known_images)
print('Encoding Complete')

# Initialize webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Failed to grab a frame.")
#         break

#     boxes, _, _ = yolov8_detector(frame)
#     combined_img = yolov8_detector.draw_detections(frame)

#     result_img = perform_face_recognition(combined_img, encode_list_known, class_names)



#     cv2.imshow("Combined Detection", result_img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# model=load_model('saved_model 20-01-22.h5')
# cap = cv2.VideoCapture('/Users/apple/Desktop/final_year_project/AI-Enabled-Student-Tracking-System/IMG_8975.MOV')
# cap = cv2.VideoCapture("./24.mp4")
cap = cv2.VideoCapture(0)
# print(cap.isOpened())
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break
    
    boxes, _, _ = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)

    result_img = perform_face_recognition(combined_img, encode_list_known, class_names)

    for i in range(10):
        ret,frame=cap.read()
        if ret == False:
            break
        image = im.resize(frame, width=1000, height=1000, inter=cv2.INTER_AREA)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)
    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)
    output=model.predict(imagedump)
    loss=mean_squared_loss(imagedump,output)
    if ret == False:
        print("video end")
        break
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    print(loss)
    if loss>0.00064:
        print('Abnormal Event Detected')
        cv2.putText(result_img,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_SIMPLEX,2,(250,24,255),4)
    cv2.imshow("video",result_img)
cap.release()
cv2.destroyAllWindows()