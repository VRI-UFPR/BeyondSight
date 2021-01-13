import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


# path = "/home/dvruiz/pCloudDrive/Cellphone/20191119_200732.jpg"
path = "/home/dvruiz/pCloudDrive/Cellphone/20191119_200733.jpg"

# img = cv2.imread(path)
img = cv2.imread(path)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([hsv],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

lower_blue = np.array([100,100,100])
upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)
res = cv2.resize(res, None, fx=0.25, fy=0.25)
cv2.imshow("output", res)
cv2.waitKey(10000)
