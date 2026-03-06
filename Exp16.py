import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide tkinter window
Tk().withdraw()

# Select image file
file_path = askopenfilename(title="Select the image")

# Read image
img = cv2.imread(file_path)

if img is None:
    print("Image not loaded")
    exit()

# Convert BGR to RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu threshold
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Morphology operation
kernel = np.ones((2,2),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)

# Dilation
sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Show results
plt.figure(figsize=(12,8))

plt.subplot(231)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(232)
plt.imshow(gray,cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(233)
plt.imshow(thresh,cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(234)
plt.imshow(closing,cmap="gray")
plt.title("Morphology Closing")
plt.axis("off")

plt.subplot(235)
plt.imshow(sure_bg,cmap="gray")
plt.title("Dilation")
plt.axis("off")

plt.tight_layout()
plt.show()
