import cv2 
import os 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'test_image.jpg')
image = cv2.imread(img_path)
cv2.imshow('result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
