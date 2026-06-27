import os
from config import ORGANIZED_GALLERY


def gallery_statistics():

    stats = {}

    total_images = 0

    if not os.path.exists(ORGANIZED_GALLERY):
        return stats, total_images

    for person in os.listdir(ORGANIZED_GALLERY):

        folder = os.path.join(ORGANIZED_GALLERY, person)

        if not os.path.isdir(folder):
            continue

        count = 0

        for file in os.listdir(folder):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                count += 1

        stats[person] = count

        total_images += count

    return stats, total_images