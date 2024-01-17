import cv2
import numpy as np
import face_recognition
from yolov8 import YOLOv8
import os
from datetime import datetime

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

def perform_face_recognition_and_mobile_detection(image):
    path = 'ImagesAttendance'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = [face_recognition.face_encodings(img)[0] for img in images]

    imgS = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            mobile_detected = mobile_phone_near_face(y1, x1, y2, x2, yolov8_detector)
            markAttendanceAndMobileBehaviour(name, mobile_detected)

    return image

def mobile_phone_near_face(face_y1, face_x1, face_y2, face_x2, yolov8_detector):
    mobile_class_id = 67  # Adjust if needed for your YOLOv8 model
    mobile_threshold = 0.2  # Adjust as needed

    boxes, _, class_ids = yolov8_detector.detect_objects(frame)

    for box, class_id in zip(boxes, class_ids):
        if class_id == mobile_class_id:
            box_y1, box_x1, box_y2, box_x2 = box

            # Check for overlap between face and mobile bounding boxes
            if (
                box_x1 < face_x2 + mobile_threshold * (face_x2 - face_x1) and
                box_x2 > face_x1 - mobile_threshold * (face_x2 - face_x1) and
                box_y1 < face_y2 + mobile_threshold * (face_y2 - face_y1) and
                box_y2 > face_y1 - mobile_threshold * (face_y2 - face_y1)
            ):
                return True  # Mobile detected near the face

    return False  # No mobile detected near the face

def markAttendanceAndMobileBehaviour(name, mobile_detected):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString},{mobile_detected}')  # Write mobile behaviour to CSV

# The rest of the code remains the same for camera capture and displaying frames


# Rest of the code remains the same for camera capture and displaying frames


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        boxes, _, class_ids = yolov8_detector(frame)  # Perform object detection
        result_img = perform_face_recognition_and_mobile_detection(frame)  # Face recognition and mobile detection
        cv2.imshow("Combined Detection", result_img)  # Display processed image

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break
    else:
        print("Error reading frame from camera")
        break

cap.release()
cv2.destroyAllWindows()