#!/usr/bin/env python3

import sys

sys.path.append("build")
import example
import cv2
import numpy as np

B = example.opencv_usage()
img = np.ones((600, 600, 3)) * (255, 0, 255)
cv2.imshow("Image from python", img)
cv2.waitKey()
cv2.destroyAllWindows()

print(B)
