from pathlib import Path
import cv2
import numpy as np
from Refocus import refocus

patch_len = 5
count = 0
a= 0
factor = 1
images = []
files = Path(r"DragonAndBunnies\DragonsAndBunnies_5x5_ap6.6").glob("*.png")
for file in files:
    img = cv2.imread(str(file))
    if count % patch_len == 0:
        images.append([])
    images[count // patch_len].append(img)
    count += 1
images.reverse()
Light_Field = np.array(images)

Final_img = refocus(Light_Field, patch_len, factor, a)
cv2.imwrite(fr"Test.jpg",Final_img)