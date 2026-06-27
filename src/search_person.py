import os

from config import ORGANIZED_GALLERY


def search_person(person_name):

    folder = os.path.join(ORGANIZED_GALLERY, person_name)

    if not os.path.exists(folder):
        return []

    images = []

    for file in os.listdir(folder):

        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(os.path.join(folder, file))

    return images