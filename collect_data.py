import cv2
import numpy as np

classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
data = []

while len(data) < 50:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        data.append(face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        print(len(data), "/50")
        break

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

name = input("Enter your name: ")
for i, img in enumerate(data):
    cv2.imwrite(f"images/{name}_{i}.jpg", img)

print("✅ Face images collected")
