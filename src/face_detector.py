import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return detector.detect_faces(rgb)