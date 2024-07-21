"""
허프 변환:

이미지에서 직선이나 원 같은 기본적인 형태를 찾는 데 사용

1. 이진화된 이미지 생성
2. 허프 공간 설정 
    - 직선의 기울기 m, y절편 b를 각각 x, y 축으로 하는 2차원 공간
3. 가능한 모든 직선들에 대해 고려 
4. 각 직선들의 교차점을 누적, 누적값이 크면 실제 직선에 해당
5. 임계값 설정하여 실제로 인식할 직선 결정
6. 파라미터 (m, b)를 통해 직선 추출 
"""

import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'test_image.jpg')
img = cv2.imread(img_path)

def Canny(img):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny 

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img


def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
    [(200, height), (1000, height), (550, 250)]
    ], dtype = np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image


lane_image = np.copy(img)
canny = Canny(lane_image)
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_img = display_lines(lane_image, lines)
combo_img = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)
cv2.imshow("result", combo_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
