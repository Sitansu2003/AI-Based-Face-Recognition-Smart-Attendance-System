import csv
import pickle
import sqlite3
from datetime import datetime
import cv2

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- LBPH MODEL ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# ---------------- LABELS ----------------
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- CSV ----------------
csv_file = open("attendance.csv", "a", newline="")
writer = csv.writer(csv_file)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("attendance.db")
cur = conn.cursor()

# ---------------- TABLES ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT UNIQUE
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    student_id TEXT,
    name TEXT,
    date TEXT,
    time TEXT,
    status TEXT,
    UNIQUE(student_id, date)
)
""")
conn.commit()

# ---------------- STUDENT ID FUNCTION ----------------
def get_or_create_student_id(name):
    cur.execute("SELECT student_id FROM students WHERE name=?", (name,))
    row = cur.fetchone()

    if row:
        return row[0]

    cur.execute("SELECT COUNT(*) FROM students")
    count = cur.fetchone()[0] + 1
    student_id = f"STD{count:03d}"

    cur.execute(
        "INSERT INTO students (student_id, name) VALUES (?,?)",
        (student_id, name)
    )
    conn.commit()
    return student_id

# ---------------- ABSENT FUNCTION ----------------
def mark_absent_students():
    today = datetime.now().strftime("%d-%m-%Y")

    cur.execute("SELECT student_id, name FROM students")
    students = cur.fetchall()

    for student_id, name in students:
        cur.execute(
            "SELECT 1 FROM attendance WHERE student_id=? AND date=?",
            (student_id, today)
        )
        exists = cur.fetchone()

        if not exists:
            cur.execute(
                "INSERT INTO attendance VALUES (?,?,?,?,?)",
                (student_id, name, today, "-", "ABSENT")
            )

    conn.commit()

# ---------------- SESSION CONTROL ----------------
marked_today = set()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        id_, distance = recognizer.predict(face)

        if distance < 60:
            name = labels[id_]
            student_id = get_or_create_student_id(name)
            color = (0, 255, 0)

            today = datetime.now().strftime("%d-%m-%Y")

            if (student_id, today) not in marked_today:
                time_now = datetime.now().strftime("%H:%M:%S")

                writer.writerow([student_id, name, today, time_now, "PRESENT"])

                try:
                    cur.execute(
                        "INSERT INTO attendance VALUES (?,?,?,?,?)",
                        (student_id, name, today, time_now, "PRESENT")
                    )
                    conn.commit()
                except sqlite3.IntegrityError:
                    pass

                marked_today.add((student_id, today))
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    
    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
mark_absent_students()
csv_file.close()
conn.close()
cv2.destroyAllWindows()
