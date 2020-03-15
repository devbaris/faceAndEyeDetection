import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)  # Getting image via Webcam
capture.set(3, 640)  # Width
capture.set(4, 480)  # Height

while True:

    image = capture.read()

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayImage, 1.3, 5)  # Faces detection

    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draws a rectangle around the face

        faceGrayRoi = grayImage[y:y + h, x:x + w]
        faceColorRoi = image[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(faceGrayRoi)  # Eyes detection

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(faceColorRoi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)  # Draws a rectangle around the eyes

    cv2.imshow('Face And Eye Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
