import os
import cv2
import shutil

from PIL import Image
import numpy as np

from face_recognizer import recognize
from config import NEW_GALLERY, ORGANIZED_GALLERY

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def organize_gallery():

    os.makedirs(ORGANIZED_GALLERY, exist_ok=True)

    total_images = 0
    organized = 0
    no_face = 0
    recognized = 0

    for filename in os.listdir(NEW_GALLERY):

        print("Found:", filename)

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            print("Skipped:", filename)
            continue

        image_path = os.path.join(NEW_GALLERY, filename)

        image = cv2.imread(image_path)

        # Pillow fallback
        if image is None:
            try:
                image = np.array(Image.open(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception:
                print(f"Cannot read: {filename}")
                continue

        print("Processing:", filename)

        total_images += 1

        results = recognize(image)

        # No face detected
        if len(results) == 0:

            folder = os.path.join(ORGANIZED_GALLERY, "No_Face")
            os.makedirs(folder, exist_ok=True)

            shutil.copy(image_path, os.path.join(folder, filename))

            organized += 1
            no_face += 1
            continue

        copied = set()

        has_known_person = False

        for result in results:

            person = result["name"]

            if person in copied:
                continue

            copied.add(person)

            if person != "Unknown":
                has_known_person = True

            folder = os.path.join(ORGANIZED_GALLERY, person)
            os.makedirs(folder, exist_ok=True)

            shutil.copy(image_path, os.path.join(folder, filename))

        organized += 1

        if has_known_person:
            recognized += 1

    return {
        "processed": total_images,
        "organized": organized,
        "recognized": recognized,
        "no_face": no_face
    }


if __name__ == "__main__":

    result = organize_gallery()

    print("\n====================================")
    print("Gallery Organized Successfully!")
    print("====================================")
    print(f"Images Processed : {result['processed']}")
    print(f"Images Organized : {result['organized']}")
    print(f"Recognized Images: {result['recognized']}")
    print(f"No Face Images   : {result['no_face']}")