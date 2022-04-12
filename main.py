from pathlib import Path
import cv2
import numpy as np
from Refocus import refocus

num_views = 5
count = 0
a= 0
factor = 1

# Reminds that Views of the light field should be arranges from top to the bottom and from left to the right
# This dataset is arragned from bottom to the top, so I reverse it
images = []
files = Path(r"DragonAndBunnies\DragonsAndBunnies_5x5_ap6.6").glob("*.png")
for file in files:
    img = cv2.imread(str(file))
    if count % num_views == 0:
        images.append([])
    images[count // num_views].append(img)
    count += 1
images.reverse()
Light_Field = np.array(images)

Final_img = refocus(Light_Field, num_views, factor, a)
cv2.imwrite(fr"Test.jpg",Final_img)