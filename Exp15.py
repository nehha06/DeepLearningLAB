import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# hide tkinter window
Tk().withdraw()

# open file picker
file_path = askopenfilename(title="Select the image")

# read image
img = cv2.imread(file_path)

if img is None:
    print("Image not loaded")
    exit()

# convert BGR to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape pixels
pixels = np.float32(rgb_img.reshape((-1,3)))

# Kmeans criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0.2)

K = 3

# apply kmeans
_, labels, centers = cv2.kmeans(pixels,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_img = centers[labels.flatten()].reshape(rgb_img.shape)

# show results
plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(122)
plt.imshow(segmented_img)
plt.title("Segmented Image")
plt.axis("off")

plt.show()
