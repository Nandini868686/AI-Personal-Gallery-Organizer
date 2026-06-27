import os
import cv2

# Path to the dataset
DATASET_PATH = "dataset/original_images"

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

total_images = 0

# Read every person's folder
for person_name in os.listdir(DATASET_PATH):

    person_folder = os.path.join(DATASET_PATH, person_name)

    # Skip if it is not a folder
    if not os.path.isdir(person_folder):
        continue

    print(f"\nPerson: {person_name}")

    # Read every image inside that folder
    for image_name in os.listdir(person_folder):

        if image_name.lower().endswith(IMAGE_EXTENSIONS):

            image_path = os.path.join(person_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_name}")
                continue

            total_images += 1

            height, width, channels = image.shape

            print(f"Image : {image_name}")
            print(f"Width : {width}")
            print(f"Height: {height}")
            print(f"Channels: {channels}")
            print("-" * 40)

print(f"\nTotal Images = {total_images}")