import cv2
import pickle
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# ------------------------------------
# Configuration
# ------------------------------------

IMAGE_PATH = "test_images/test1.jpg"
DATABASE_PATH = "models/face_database.pkl"

THRESHOLD = 0.70

# ------------------------------------
# Load Database
# ------------------------------------

with open(DATABASE_PATH, "rb") as f:
    database = pickle.load(f)

detector = MTCNN()
embedder = FaceNet()

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found.")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

faces = detector.detect_faces(rgb)

print(f"Detected {len(faces)} face(s)\n")

for face in faces:

    x, y, w, h = face["box"]

    x = max(0, x)
    y = max(0, y)

    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        continue

    crop = cv2.resize(crop, (160,160))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    embedding = embedder.embeddings([crop])[0]

    best_person = "Unknown"
    best_score = 0

    for person, db_embedding in database.items():

        score = 1 - cosine(embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_person = person

    if best_score < THRESHOLD:
        best_person = "Unknown"

    color = (0,255,0)

    if best_person == "Unknown":
        color = (0,0,255)

    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)

    text = f"{best_person} ({best_score*100:.1f}%)"

    cv2.putText(
        image,
        text,
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    print(text)

cv2.imshow("AI Personal Gallery Organizer",image)
cv2.waitKey(0)
cv2.destroyAllWindows()