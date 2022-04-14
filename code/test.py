#%%
# IMPORT
from matplotlib.pyplot import axis
import numpy as np
import cv2
# %%
img1 = cv2.imread('../output_data/00239040.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../output_data/00239041.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('../output_data/00239042.png', cv2.IMREAD_GRAYSCALE)
# %%
test = np.dstack((img1,img2))
# %%
test = np.dstack((test, img3))
# %%

v1 = np.ones((10,20))
v2 = np.ones((10,20))*2
v3 = np.ones((10,20))*3
# %%
