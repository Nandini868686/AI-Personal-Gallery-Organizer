import pickle
import cv2

from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

from face_detector import detect_faces
from config import DATABASE_PATH, THRESHOLD

embedder = FaceNet()

with open(DATABASE_PATH, "rb") as f:
    database = pickle.load(f)


def recognize(image):

    faces = detect_faces(image)

    results = []

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

        best_name = "Unknown"
        best_score = 0

        for person, db_embedding in database.items():

            score = 1 - cosine(embedding, db_embedding)

            if score > best_score:

                best_score = score
                best_name = person

        if best_score < THRESHOLD:
            best_name = "Unknown"

        results.append({

            "name": best_name,
            "score": best_score,
            "box": (x,y,w,h)

        })

    return results
    