import os
import pickle
import numpy as np
from keras_facenet import FaceNet
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
FACES_PATH = "output/detected_faces"
OUTPUT_FILE = "models/embeddings.pkl"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load FaceNet model
embedder = FaceNet()

embeddings = []
labels = []

print("=" * 60)
print("Generating Face Embeddings")
print("=" * 60)

for person_name in sorted(os.listdir(FACES_PATH)):

    person_folder = os.path.join(FACES_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Processing: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((160, 160))

            image = np.asarray(image)

            embedding = embedder.embeddings([image])[0]

            embeddings.append(embedding)
            labels.append(person_name)

        except Exception as e:
            print(f"Skipped: {image_name}")
            print(e)

# Save embeddings
data = {
    "embeddings": embeddings,
    "labels": labels
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("\n" + "=" * 60)
print("Embeddings Generated Successfully!")
print("=" * 60)
print(f"Total Embeddings : {len(embeddings)}")
print(f"Saved to : {OUTPUT_FILE}")