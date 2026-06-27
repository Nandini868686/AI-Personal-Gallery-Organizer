import pickle
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = "models/embeddings.pkl"
OUTPUT_FILE = "models/face_database.pkl"

# Load embeddings
with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
labels = data["labels"]

database = {}

# Group embeddings by person
for embedding, label in zip(embeddings, labels):

    if label not in database:
        database[label] = []

    database[label].append(embedding)

# Compute average embedding for each person
face_database = {}

for person in sorted(database.keys()):

    avg_embedding = np.mean(database[person], axis=0)

    face_database[person] = avg_embedding

    print(f"{person:<25} {len(database[person])} images")

# Save
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(face_database, f)

print("\n===================================")
print("Face Database Created Successfully!")
print("===================================")

print(f"Total Persons : {len(face_database)}")
print(f"Saved to      : {OUTPUT_FILE}")