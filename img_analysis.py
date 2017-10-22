import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

 
TRAINING_CSV_PATH = '../data/my_training_data'
TRAINING_CSV_FILE = 'driving_log.csv'
TRAINING_IMAGE_PATH = os.path.join(TRAINING_CSV_PATH, 'IMG')


def load_data():
    colnames = ['center_img', 'left_img', 'right_img', 'steering', 'throttle', 'break', 'speed']
    csvdata = pd.read_csv(os.path.join(TRAINING_CSV_PATH, TRAINING_CSV_FILE), names = colnames)
    remap = lambda n: os.path.join(TRAINING_IMAGE_PATH, os.path.basename(n))
    csvdata.iloc[:,0:3] = csvdata.iloc[:,0:3].applymap(remap)
    return csvdata

csvdata = load_data()

# Load originals and convert them to RGB
idx = np.random.randint(csvdata.shape[0]+1)
to_rgb = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ctr = to_rgb(cv2.imread(csvdata.iloc[idx][0]))
img_lft = to_rgb(cv2.imread(csvdata.iloc[idx][1]))
img_rht = to_rgb(cv2.imread(csvdata.iloc[idx][2]))

# Crop images
img_height = img_ctr.shape[0]
img_ctr_cropped = img_ctr[60:img_height-25, :, :]
img_lft_cropped = img_lft[60:img_height-25, :, :]
img_rht_cropped = img_rht[60:img_height-25, :, :]

# Convert to grayscale
to_gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_ctr_gray = to_gray(img_ctr_cropped)
img_lft_gray = to_gray(img_lft_cropped)
img_rht_gray = to_gray(img_rht_cropped)

# Plot 3 original images in 1st row
plt.subplot(331)
plt.imshow(img_lft)
plt.title('Original Left')

plt.subplot(332)
plt.imshow(img_ctr)
plt.title('Original Center')

plt.subplot(333)
plt.imshow(img_rht)
plt.title('Original Right')

# Plot 3 cropped images in the 2nd row
plt.subplot(334)
plt.imshow(img_lft_cropped)
plt.title('Cropped Left')

plt.subplot(335)
plt.imshow(img_ctr_cropped)
plt.title('Cropped Center')

plt.subplot(336)
plt.imshow(img_rht_cropped)
plt.title('Cropped Right')

# Plot 3 gray images in the 3nd row
plt.subplot(337)
plt.imshow(img_lft_gray, cmap='gray')
plt.title('Gray Left')

plt.subplot(338)
plt.imshow(img_ctr_gray, cmap='gray')
plt.title('Gray Center')

plt.subplot(339)
plt.imshow(img_rht_gray, cmap='gray')
plt.title('Gray Right')

# Show the plot
plt.tight_layout()
plt.show()











