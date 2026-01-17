import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

faces = []
labels = []
label_map = {}
label_id = 0

for person in os.listdir(dataset_path):
    label_map[label_id] = person
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))

        faces.append(img)
        labels.append(label_id)

    label_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("face_model.yml")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Model trained and labels saved")
