import cv2
import numpy as np
from image_object import ImageObject
from image_processing import *
import math

# Parameters
TARGET_IMAGE_SIZE = 540
OBJECT_UPP_BOUND_SCREEN_SIZE = 0.8
OBJECT_LOW_BOUND_SIZE = 800
SQUARE_WIDTH_HEIGHT_THRESHOLD = 0.15
CIRCLE_WIDTH_HEIGHT_THRESHOLD = 0.15
CIRCLE_RATIO_MATCH_THRESHOLD = 0.15

# Macros
def load_image(image_directory,image_file_name):
    return cv2.imread(f'{image_directory}/{image_file_name}')

def process(image_directory, image_file_name):
    # Load image
    img = load_image(image_directory,image_file_name)
    # Scale image
    img = scale(img, TARGET_IMAGE_SIZE)
    # Add border to the scaled image
    img = addborder(img, 100)
    # Denoise image
    img_denoised = denoise(img, 15)
    # Sharpen image
    img_sharpen = sharpen(img_denoised)
    # Grayscale image
    img_gray = gray(img_sharpen)
    # Gaussian Blur image
    img_blur = gaussianblur(img_gray,5)
    # Threshold
    threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 5)

    # First contouring
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    high_bound = (img.shape[0]*img.shape[1]) * OBJECT_UPP_BOUND_SCREEN_SIZE
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > OBJECT_LOW_BOUND_SIZE and area < high_bound:
            filtered_contours.append(cnt)

    cv2.drawContours(threshold, filtered_contours, -1, (0, 0, 0), 2)

    for cnt in filtered_contours:
            cv2.fillPoly(threshold, [cnt], color=(0,0,0))

    # Second contouring
    img_blur_contour = gaussianblur(threshold, 9)
    fill_contour = cv2.findContours(img_blur_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    detected_object = []
    index = 1
    for cnt in fill_contour:
        area = cv2.contourArea(cnt)
        if area > OBJECT_LOW_BOUND_SIZE and area < high_bound:
            fill_contour, y, w, h = cv2.boundingRect(cnt)
            corners = len(cv2.approxPolyDP(cnt, 6.7, True))
            hull = cv2.convexHull(cnt)
            corners_hull = len(cv2.approxPolyDP(hull, 6.5, True))
            obj = ImageObject([fill_contour, y, w, h], index,area,cnt)
            if corners == corners_hull:
                obj.corner = corners
            else:
                obj.corner = -1
            detected_object.append(obj)
            index += 1

    triangular_shape = 0
    rectangle_shape = 0
    square_shape = 0
    pentagon_shape = 0
    hexagon_shape = 0
    circle_shape = 0
    others_shape = 0

    for object in detected_object:
        cv2.rectangle(img, (object.postion[0], object.postion[1]), (object.postion[0] + object.postion[2], object.postion[1] + object.postion[3]), (255, 0, 0), 2)
        cv2.putText(img, str(object.id), (object.postion[0],object.postion[1]-5), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),2)
        if object.corner == 3:
            cv2.putText(img,"Triangle", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
            triangular_shape += 1
        elif object.corner == 4:
            widthHeightRatio = object.postion[2] / object.postion[3]
            if (widthHeightRatio > (1 - SQUARE_WIDTH_HEIGHT_THRESHOLD) and widthHeightRatio < (1 + SQUARE_WIDTH_HEIGHT_THRESHOLD)):
                cv2.putText(img, "Square", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 0, 0), 2)
                square_shape += 1
            else:
                cv2.putText(img, "Rectangle", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
                rectangle_shape += 1
        elif object.corner == 5:
            cv2.putText(img, "Pentagon", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
            pentagon_shape += 1
        elif object.corner == 6:
            cv2.putText(img, "Hexagon", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 0, 0), 2)
            hexagon_shape += 1
        else:
            # Check if the shape is circle
            objectArea = object.area
            circleRadius = object.postion[2]/2
            circleArea = math.pi * (circleRadius)**2
            circleMatchRatio = objectArea/ circleArea
            widthHeightRatio = object.postion[2] / object.postion[3]
            if (circleMatchRatio > (1-CIRCLE_RATIO_MATCH_THRESHOLD) and circleMatchRatio < (1+CIRCLE_RATIO_MATCH_THRESHOLD) and widthHeightRatio > (1 - CIRCLE_WIDTH_HEIGHT_THRESHOLD) and widthHeightRatio < (1 + CIRCLE_WIDTH_HEIGHT_THRESHOLD)):
                cv2.putText(img, "Circle", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 0, 0), 2)
                circle_shape += 1
            else:
                cv2.putText(img, "Undefined Shape", (object.postion[0] + 30, object.postion[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 0, 0), 2)
                others_shape += 1

    detected_object.sort(key=lambda object: object.area)
    cv2.imshow(image_file_name, img)
    print(f'\n\n\n------ {image_file_name} stats ------')
    print(f'The number of objects in the scene is : {len(detected_object)}')
    print(f'The smallest object in the scene is : {detected_object[0].id}')
    print(f'The largest object in the scene is : {detected_object[-1].id}')
    print(f'The number of triangular-like object in the scene is : {triangular_shape}')
    print(f'The number of rectangle-like object in the scene is : {rectangle_shape}')
    print(f'The number of square-like object in the scene is : {square_shape}')
    print(f'The number of hexagon-like object in the scene is : {hexagon_shape}')
    print(f'The number of circle-like object in the scene is : {circle_shape}')
    print(f'The number of object are unidentified with the shape is : {others_shape}')
    print(f'------ end of {image_file_name} stats ------')

    cv2.waitKey(0)