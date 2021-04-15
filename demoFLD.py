import cv2
import dlib

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    success, frame = cap.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(grayImg)
    for face in faces:

        face_landmarks = dlib_facelandmark(grayImg, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()