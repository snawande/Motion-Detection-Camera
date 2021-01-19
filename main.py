# imports
import os,sys,time

import numpy as np

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

import cv2

import matplotlib.pyplot as plt

time_record = []
motion_record = []
# initialize frame capture and first frame
webcam = cv2.VideoCapture(0)
first_frame = None
start_time = time.time()
while True:
    # check frame status
    (check,original_frame) = webcam.read()
    # if the frame sttus is not true, then break loop
    if check != True:
        break

    # make a grayscale frame image
    grayscale_frame = cv2.cvtColor(original_frame,cv2.COLOR_BGR2GRAY)
    # apply blur to fine out the details of the grayscale image
    new_grayscale_frame = cv2.GaussianBlur(grayscale_frame,(15,15),0)

    # initialize first_frame
    if first_frame is None:
        first_frame = new_grayscale_frame
        continue

    # delta frame
    delta_frame = cv2.absdiff(first_frame,new_grayscale_frame)

    # threshold frame
    threshold_frame = cv2.threshold(delta_frame,15,255,cv2.THRESH_BINARY)[1]

    # morpholigical transformations, retrived from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # erodes picture by a factor of 4 first to keep a small outline of needed pixels
    # effectively dilates the thresholded frame to amplify pixels, by a factor of 7

    kernel = np.ones((2,2),np.uint8)
    erode = cv2.erode(threshold_frame,kernel,iterations=4)
    eroded_dailted = cv2.dilate(erode,kernel,iterations=8)

    # find contours on thresholded image
    (contours,_) = cv2.findContours(eroded_dailted.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # make coutour frame
    countours_frame = original_frame.copy()
    # contours that need to be targeted
    targets = []
    # loop over the contours
    for c in contours:
        # if the contour area is less than 500 pixels, then ignore it
        if cv2.contourArea(c) < 500:
                continue

        # M_data will store all the data of a contour's movement
        Movements = cv2.moments(c)

        # cx and cy calculate the centroid-coordinates for the countour.
        # retrived from:
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
        cx = int(Movements['m10']/Movements['m00'])
        cy = int(Movements['m01']/Movements['m00'])

        # Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        x,y,w,h = cv2.boundingRect(c)
        rx = x + int(w/2)
        ry = y + int(h/2)
        ca = cv2.contourArea(c)

        # plotting contours on the countours_frame
        cv2.drawContours(countours_frame,[c],0,(0,0,0),2)
        cv2.rectangle(countours_frame,(x,y),(x+w,y+h),(0,255,0),2)
        # save target contours
        targets.append((cx,cy,ca))


    # getting the center_coordinates for the contour with the largest area
    mx = 0
    my = 0
    area = 0

    for x,y,a in targets:
        if a > area:
            mx = x
            my = y
            area = a
    # plot target
    target_radius = 100
    target_frame = original_frame.copy()
    color = (51,255,51)
    center_coordinates = (mx,my)
    circle_thickness = 5
    line_thickness = 2


    if targets:
        cv2.circle(target_frame, center_coordinates,target_radius,(51,255,51),circle_thickness)
        cv2.line(target_frame,(mx-target_radius,my),(mx+target_radius,my),(51,255,51),line_thickness)
        cv2.line(target_frame,(mx,my-target_radius),(mx,my+target_radius),(51,255,51),line_thickness)
        elapsed_time = time.time()-start_time
        time_record.append(elapsed_time)
        motion_record.append(1)

    else:
        elapsed_time = time.time()-start_time
        time_record.append(elapsed_time)
        motion_record.append(0)

        #print('no motion detected')

    # update first_frame with each iteration
    first_frame = new_grayscale_frame

    # display the target frame
    cv2.imshow("Detecting Object in Motion",target_frame)

    # quit the program if 'q' key is pressed on the keyboard
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# release webcam
webcam.release()
# close all windows
cv2.destroyAllWindows()


plt.plot(time_record, motion_record)
plt.xlabel('Time (seconds)')
plt.ylabel('Mtion Status')
plt.title('Motion Graph')
plt.show()















