import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


path = "/home/dvruiz/pCloudDrive/Cellphone/20191119_200732.jpg"

img = cv2.imread(path)
# img = cv2.imread(path, cv2.COLOR_BGR2LAB)

b, g, r   = img[:, :, 0], img[:, :, 1], img[:, :, 2] # For RGB image

histB = cv2.calcHist([b],[0],None,[256],[0,256])
cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

histG = cv2.calcHist([g],[0],None,[256],[0,256])
cv2.normalize(histG, histG, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

histR = cv2.calcHist([r],[0],None,[256],[0,256])
cv2.normalize(histR, histR, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

histB = histB.flatten()
histG = histG.flatten()
histR = histR.flatten()

# histB = savgol_filter(histB, 51, 3) # window size 51, polynomial order 3
# histG = savgol_filter(histG, 51, 3) # window size 51, polynomial order 3
# histR = savgol_filter(histR, 51, 3) # window size 51, polynomial order 3

hist = histB - histG - histR

hist = savgol_filter(hist, 51, 3) # window size 51, polynomial order 3

plt.plot(hist,color = 'b')
plt.xlim([0,256])
plt.show()


indices = find_peaks(hist)[0]

histPeaks = hist[indices]
mymap = np.argsort(histPeaks)

pos = indices[mymap]

# low  = pos[pos.shape[0]-2]
# high = pos[pos.shape[0]-1]
high = pos[pos.shape[0]-2]

# low2 = min(low,high)
# high2 = max(low,high)

low2 = high-32
high2 = high+32

g = np.where((b > low2)and(b < high2), 255, 0)
r = np.where((b > low2)and(b < high2), 255, 0)

rgb = np.dstack((r,g,b))
# print(low2,high2)
# #
# ret,b = cv2.threshold(b,low2,high2,cv2.THRESH_BINARY)
# #
b = cv2.resize(rgb, None, fx=0.1, fy=0.1)
cv2.imshow("output", b)
cv2.waitKey(10000)
