import cv2 
import os 
import numpy as np 

cur_dir = os.getcwd()
img_path = os.path.join(cur_dir, 'Self-Driving Car Deep Course', 'Image', 'test_image.jpg')
img = cv2.imread(img_path)

# Calculates the x and y coordinates of the start and end points of a line, given the slope and intercept.
def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# Separates the detected lines into left and right based on their slope, then averages them to create a single line for each side.
def average_slope_intercept(img, lines):
    left_fit = [] 
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    left_line = right_line = None
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(img, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(img, right_fit_average)

    return np.array([line for line in [left_line, right_line] if line is not None])

# Applies Canny edge detection to the image.
def Canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny 

# Draws lines on a blank image.
def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope = (y2 - y1) / (x2 - x1)
            color = (0, 255, 255) if slope < 0 else (255, 255, 255) # Yellow for central line, white for outer line
            cv2.line(line_img, (x1, y1), (x2, y2), color, 10)
    return line_img

# Fills the area between the two lane lines with a light green color.
def fill_road(img, lines):
    road_img = np.zeros_like(img)
    if len(lines) == 2:
        left_line, right_line = lines
        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)
        cv2.fillPoly(road_img, [pts], (144, 238, 144))  # Light green color
    return road_img

# Masks the image to only consider a region of interest, typically the road area.
def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
        [(200, height), (1000, height), (550, 250)]
    ], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cap = cv2.VideoCapture("C:\\Users\\dknjy\\.anaconda\\autonomous\\Self-Driving Car Deep Course\\Video\\test2.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    canny_img = Canny(frame)
    cropped_image = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    
    line_img = display_lines(frame, averaged_lines)
    road_img = fill_road(frame, averaged_lines)
    
    combo_img = cv2.addWeighted(frame, 0.8, road_img, 0.5, 1)
    combo_img = cv2.addWeighted(combo_img, 1, line_img, 1, 1)

    cv2.imshow("result", combo_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
