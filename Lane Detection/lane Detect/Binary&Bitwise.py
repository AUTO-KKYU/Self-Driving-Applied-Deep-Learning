import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'test_image.jpg')
img = cv2.imread(img_path)
lane_image = np.copy(img)

def Canny(img):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny 


def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
    [(200, height), (1000, height), (550, 250)]
    ], dtype = np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


canny = Canny(lane_image)
cropped_image = region_of_interest(canny)
cv2.imshow("result", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(canny)
# plt.show()
