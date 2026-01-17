import cv2
import pyttsx3
import os
import time
import math

# ================= VOICE SETUP =================
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ================= INPUT =================
name = input("Enter person name: ").strip()

dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)
os.makedirs(person_path, exist_ok=True)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opened")
    exit()

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= CONFIG =================
TOTAL_IMAGES = 30
count = 0

STATE = "WAIT"
last_spoken_state = None
countdown_start = None
COUNTDOWN_SECONDS = 5
CIRCLE_RADIUS = 120

# Speak first instruction
speak("Press S to submit")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    #  default circle color
    circle_color = (0, 0, 255)

    # -------- Blur outside circle --------
    blurred = cv2.GaussianBlur(frame, (35, 35), 0)
    mask = frame.copy()
    mask[:] = 0
    cv2.circle(mask, (cx, cy), CIRCLE_RADIUS, (255, 255, 255), -1)
    frame = cv2.bitwise_and(frame, mask) + cv2.bitwise_and(
        blurred, cv2.bitwise_not(mask)
    )

    # -------- VOICE ON STATE CHANGE --------
    if STATE != last_spoken_state:
        if STATE == "COUNTDOWN":
            speak("Please wait. Capturing will start in five seconds")
        elif STATE == "CAPTURE":
            speak("Capturing face images")
        last_spoken_state = STATE

    # -------- STATE LOGIC --------
    if STATE == "WAIT":
        cv2.putText(frame, "Press S to Submit", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    elif STATE == "COUNTDOWN":
        elapsed = int(time.time() - countdown_start)
        remaining = COUNTDOWN_SECONDS - elapsed

        if remaining > 0:
            cv2.putText(frame, f"Please wait... {remaining}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            STATE = "CAPTURE"

    elif STATE == "CAPTURE":
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, fw, fh) in faces:
            fx = x + fw // 2
            fy = y + fh // 2
            dist = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)

            # face alignment check
            if dist < CIRCLE_RADIUS :
                circle_color = (0, 255, 0)

                face = gray[y:y+fh, x:x+fw]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(f"{person_path}/{count}.jpg", face)
                count += 1
                time.sleep(0.2)
                break

        cv2.putText(frame, f"Captured {count}/{TOTAL_IMAGES}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        if count >= TOTAL_IMAGES:
            speak("Face registration completed successfully")
            break

    #  DRAW CIRCLE (AFTER LOGIC)
    cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, circle_color, 3)

    # OPTIONAL TEXT
    if circle_color == (0, 255, 0):
        cv2.putText(frame, "Face Aligned", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Align face inside circle", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and STATE == "WAIT":
        STATE = "COUNTDOWN"
        countdown_start = time.time()
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face registration completed")
