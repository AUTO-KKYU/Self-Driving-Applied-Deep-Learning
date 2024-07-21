import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'test_image.jpg')
img = cv2.imread(img_path)

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    # print(img.shape)
    y1 = img.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    left_fit = [] 
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # print(left_fit_average)
    # print(right_fit_average)
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])



def Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny 

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope = (y2 - y1) / (x2 - x1)
            # 경사를 기준으로 중앙선과 바깥선을 구분하여 색상 지정
            if slope < 0:
                color = (0, 255, 255)  # 노란색 (중앙선)
            else:
                color = (255, 255, 255)  # 흰색 (바깥선)
            cv2.line(line_img, (x1, y1), (x2, y2), color, 10)
    return line_img



def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
    [(200, height), (1000, height), (550, 250)]
    ], dtype = np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


lane_image = np.copy(img)
canny_img = Canny(lane_image)
cropped_image = region_of_interest(canny_img)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_img = display_lines(lane_image, averaged_lines)
combo_img = cv2.addWeighted(lane_image, 0.8, line_img, 1, 1)
cv2.imshow("result", combo_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
