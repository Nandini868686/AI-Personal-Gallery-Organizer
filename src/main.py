import cv2

from face_recognizer import recognize
from config import TEST_IMAGE

image = cv2.imread(TEST_IMAGE)

results = recognize(image)

for person in results:

    x,y,w,h = person["box"]

    name = person["name"]

    score = person["score"]

    color = (0,255,0)

    if name=="Unknown":

        color=(0,0,255)

    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)

    cv2.putText(

        image,

        f"{name} ({score*100:.1f}%)",

        (x,y-10),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        color,

        2

    )

cv2.imshow("AI Gallery Organizer",image)

cv2.waitKey(0)

cv2.destroyAllWindows()