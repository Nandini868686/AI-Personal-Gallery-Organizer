import cv2
from mtcnn import MTCNN

# Load the detector
detector = MTCNN()

# Image path
image_path = "dataset/original_images/Akshay Kumar/Akshay Kumar_0.jpg"

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
faces = detector.detect_faces(rgb_image)

print(f"Faces Detected: {len(faces)}")

# Draw bounding boxes
for face in faces:
    x, y, width, height = face['box']

    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

# Show image
cv2.imshow("Detected Faces", image)

cv2.waitKey(0)
cv2.destroyAllWindows()