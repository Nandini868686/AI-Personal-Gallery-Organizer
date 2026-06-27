import os
import cv2
from mtcnn import MTCNN

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_PATH = "dataset/original_images"
OUTPUT_PATH = "output/detected_faces"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

MAX_IMAGE_SIZE = 800      # Resize large images
MIN_FACE_SIZE = 40        # Ignore tiny detections

# =====================================================
# INITIALIZE DETECTOR
# =====================================================

detector = MTCNN()

total_images = 0
total_faces = 0
skipped_images = 0

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("AI PERSONAL GALLERY ORGANIZER")
print("Face Detection using MTCNN")
print("=" * 60)


# =====================================================
# PROCESS EACH PERSON
# =====================================================

for person_name in sorted(os.listdir(INPUT_PATH)):

    person_folder = os.path.join(INPUT_PATH, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"\nProcessing Person : {person_name}")

    output_person_folder = os.path.join(
        OUTPUT_PATH,
        person_name
    )

    os.makedirs(output_person_folder, exist_ok=True)

    face_number = 1

    # ---------------------------------------------

    for image_name in sorted(os.listdir(person_folder)):

        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            skipped_images += 1
            continue

        total_images += 1

        # -----------------------------------------
        # Resize huge images
        # -----------------------------------------

        h, w = image.shape[:2]

        if max(h, w) > MAX_IMAGE_SIZE:

            scale = MAX_IMAGE_SIZE / max(h, w)

            new_w = int(w * scale)
            new_h = int(h * scale)

            image = cv2.resize(image, (new_w, new_h))

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -----------------------------------------
        # Detect Faces
        # -----------------------------------------

        try:
            faces = detector.detect_faces(rgb)

        except Exception as e:

            print(f"Skipped : {image_name}")
            print(e)

            skipped_images += 1
            continue

        # -----------------------------------------
        # Save Faces
        # -----------------------------------------

        for face in faces:

            x, y, width, height = face["box"]

            x = max(0, x)
            y = max(0, y)

            width = min(width, image.shape[1] - x)
            height = min(height, image.shape[0] - y)

            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue

            cropped = image[y:y + height, x:x + width]

            if cropped.size == 0:
                continue

            save_path = os.path.join(
                output_person_folder,
                f"face_{face_number:03}.jpg"
            )

            cv2.imwrite(save_path, cropped)

            face_number += 1
            total_faces += 1

    print(f"Completed : {person_name}")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)

print(f"Images Processed : {total_images}")
print(f"Faces Saved      : {total_faces}")
print(f"Images Skipped   : {skipped_images}")