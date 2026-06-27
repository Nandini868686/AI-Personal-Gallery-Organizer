import os
import cv2
import pickle
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# -----------------------------
# Paths
# -----------------------------
IMAGE_PATH = "test_images/test1.jpg"
DATABASE_PATH = "models/face_database.pkl"

# Load face database
with open(DATABASE_PATH, "rb") as f:
    face_database = pickle.load(f)

# Initialize models
detector = MTCNN()
embedder = FaceNet()

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faces = detector.detect_faces(rgb)

if len(faces) == 0:
    print("No face detected!")
    exit()

# Use the first detected face
x, y, w, h = faces[0]["box"]

x = max(0, x)
y = max(0, y)

face = image[y:y+h, x:x+w]

face = cv2.resize(face, (160, 160))
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

embedding = embedder.embeddings([face])[0]

best_person = None
best_similarity = -1

# Compare with database
for person, db_embedding in face_database.items():

    similarity = 1 - cosine(embedding, db_embedding)

    if similarity > best_similarity:
        best_similarity = similarity
        best_person = person

print("\n==============================")
print("Recognition Result")
print("==============================")

# Threshold
if best_similarity > 0.70:
    print(f"Person     : {best_person}")
    print(f"Confidence : {best_similarity*100:.2f}%")
else:
    print("Unknown Person")