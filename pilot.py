import os

import numpy as np
import cv2
import imutils
from PIL import ImageGrab
import keyboard

x = 850
y = 1000
w = 250
h = 325

marker_color = ([150, 150, 75], [255, 255, 200]) #BRG lower to upper limit for the marker (mine is yellowish, RGB 201, 178, 108)
# default is sorta cyan (RGB 139, 221, 233)

lower = np.array(marker_color[0], dtype = "uint8")
upper = np.array(marker_color[1], dtype = "uint8")

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#cap = cv2.VideoCapture('video.mp4')

l_key = 'a'
r_key = 'd'
u_key = 's'
d_key = 'w'

pixel_deadzone = 3

def diff_to_commands(diff, filled):
    '''Take the pixel difference and translate it to the keys to press'''
    # +x is to the right, +y is below
    # not filled basically reverses the motions
    go_right = diff[0] > pixel_deadzone
    go_left = diff[0] < -pixel_deadzone
    go_down = diff[1] > pixel_deadzone
    go_up = diff[1] < -pixel_deadzone

    if not filled:
        if go_right or go_left:
            go_right = not go_right
            go_left = not go_left
        # if go_down or go_up:
        #     go_down = not go_down
        #     go_up = not go_up
        if not go_up and not go_down and not go_left and not go_right:
            go_up = True

    debug_str = ''
    if go_right: debug_str += 'r'
    if go_left: debug_str += 'l'
    if go_down: debug_str += 'd'
    if go_up: debug_str += 'u'
    print(debug_str)

    return (go_right, go_left, go_down, go_up)

def execute_commands(right, left, down, up):
    if right: keyboard.press(r_key)
    if left: keyboard.press(l_key)
    if down: keyboard.press(d_key)
    if up: keyboard.press(u_key)

    # if not right and keyboard.is_pressed(r_key): keyboard.release(r_key)
    # if not left and keyboard.is_pressed(l_key): keyboard.release(l_key)
    # if not down and keyboard.is_pressed(d_key): keyboard.release(d_key)
    # if not up and keyboard.is_pressed(u_key): keyboard.release(u_key)

    if not right: keyboard.release(r_key)
    if not left: keyboard.release(l_key)
    if not down: keyboard.release(d_key)
    if not up: keyboard.release(u_key)

marker_mask = None

lx = x
ly = y
lx2 = x+w
ly2 = y+h

no_marker_frames = 0

while True:
    #ret, img_orig = cap.read()
    #if not ret: break
    img_raw = ImageGrab.grab(bbox=(lx,ly,lx2,ly2))#.convert('RGB') #bbox specifies specific region (bbox= x,y,width,height)
    img_orig = np.array(img_raw)
    #img_orig = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

    #img = cv2.imread('unfilled_offset.jpg')
    #img = img_orig.copy()
    ball = img_orig.copy()#img[y:y+h, x:x+w]
    ball_copy = ball.copy()
    ball_gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY) 
    #ball_gray = cv2.GaussianBlur(ball_gray, (3, 3), 0)

    # Do a tophat and a threshold to convert to binary image. Tophat pulls more light colors out
    ball_gray = cv2.morphologyEx(ball_gray, cv2.MORPH_TOPHAT, rectKernel)
    ball_gray = cv2.threshold(ball_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Find contours
    cnts = cv2.findContours(ball_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    ball_center = None
    # sort by size, largest first
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        (ex, ey, ew, eh) = cv2.boundingRect(c)
        ratio = ew/eh
        if ratio > 0.9 and ratio < 1.1 and ew > 50 and eh > 50: #must be roughly 1.0 aspect ratio and minimum of 50 pixels
            ball_center = (int(ex + ew/2), int(ey + eh/2))
            cv2.rectangle(ball, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)

            #found, save this location to check next frame
            lx = lx+ex - 5
            ly = ly+ey - 5
            lx2 = lx+ex+ew + 5
            ly2 = ly+ey+eh + 5
            break
    if ball_center: #found gauge/ball
        # marker identification
        ball_copy = ball_copy[ey:ey+eh, ex:ex+ew] #must be within the gauge ball

        marker_mask = cv2.inRange(ball_copy, lower, upper)
        #marker = cv2.bitwise_and(ball, ball, mask = marker_mask) # use to show original image with mask
        # get rect for marker
        marker_mask = cv2.dilate(marker_mask, None, iterations=1)
        cnts = cv2.findContours(marker_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        marker_center = None
        # sort by size, largest first
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            (mx, my, mw, mh) = cv2.boundingRect(c)
            ratio = mw/mh
            if ratio > 0.8 and ratio < 1.2: #must be roughly 1.0 aspect ratio
                marker_center = (int(mx + mw/2 + ex), int(my + mh/2 + ey))
                cv2.rectangle(ball, (mx+ex, my+ey), (mx+mw+ex, my+mh+ey), (0, 255, 0), 1)
                break
        if marker_center:
            no_marker_frames = 0
            filled = marker_mask[marker_center[1]-ey, marker_center[0]-ex] > 0
            # print(f'Is filled? {filled}')

            diff = np.subtract(marker_center, ball_center)
            print(f'Diff is {diff[0]}x by {diff[1]}y. Filled? {filled}')
            # +x is to the right, +y is below

            (go_right, go_left, go_down, go_up) = diff_to_commands(diff, filled)
            execute_commands(go_right, go_left, go_down, go_up)
        else:
            print('Did not find marker this frame')
            no_marker_frames += 1
            if no_marker_frames > 10:
                no_marker_frames = 0
                lx = x
                ly = y
                lx2 = x+w
                ly2 = y+h
    else: # not found
        print('Did not find gauge this frame')
        lx = x
        ly = y
        lx2 = x+w
        ly2 = y+h

    cv2.imshow('ball', ball)
    cv2.imshow("gray", ball_gray)
    if marker_mask is not None: cv2.imshow("marker", marker_mask)
    cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()