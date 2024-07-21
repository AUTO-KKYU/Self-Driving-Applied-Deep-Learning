import cv2 
import os 
import numpy as np 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'road.jpg')
img = cv2.imread(img_path)
lane_image = np.copy(img)
gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
cv2.imshow('result', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()