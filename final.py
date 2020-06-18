import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx
mouse = Controller()


app = wx.App(False)
(sx, sy) = wx.GetDisplaySize()

lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([170, 120, 70])
upper_red = np.array([180, 255, 255])
imw = 500
imh = int(imw * (sy / sx))
mousePress = False
cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    img = cv2.resize(img, (imw, imh))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mix = cv2.bitwise_or(red_mask, blue_mask)
    mix = cv2.GaussianBlur(mix, (7, 7), 0)
    _, mix = cv2.threshold(mix, 50, 255, cv2.THRESH_BINARY)

    red_blur = cv2.medianBlur(red_mask, 11)
    blue_blur = cv2.medianBlur(blue_mask, 11)
    mix_blur = cv2.medianBlur(mix, 11)

    mix_cont, heir = cv2.findContours(
        mix_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_cont, heir = cv2.findContours(
        red_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_cont, heir = cv2.findContours(
        blue_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(mix_cont) == 2 and len(red_cont) == 1 and len(blue_cont) == 1:
        red_hull = cv2.convexHull(red_cont[0], False)
        img = cv2.drawContours(img, [red_hull], -1, (0, 0, 255), 1)

        blue_hull = cv2.convexHull(blue_cont[0], False)
        img = cv2.drawContours(img, [blue_hull], -1, (255, 0, 0), 1)

        M_red = cv2.moments(red_hull)
        flag1 = False

        if mousePress:
            mouse.release(Button.left)
            mousePress = False

        if M_red["m00"] != 0:
            flag1 = True
            c1X = int(M_red["m10"] / M_red["m00"])
            c1Y = int(M_red["m01"] / M_red["m00"])
            img = cv2.circle(img, (c1X, c1Y), 3, (0, 0, 255), -1)

        M_blue = cv2.moments(blue_hull)
        flag2 = False
        if M_blue["m00"] != 0:
            flag2 = True
            c2X = int(M_blue["m10"] / M_blue["m00"])
            c2Y = int(M_blue["m01"] / M_blue["m00"])
            img = cv2.circle(img, (c2X, c2Y), 3, (255, 0, 0), -1)

        if flag1 and flag2:
            cX = int((c1X+c2X)/2)
            cY = int((c1Y+c2Y)/2)
            mouseLoc = (sx - (sx * (cX/imw)), sy*(cY/imh))
            mouse.position = mouseLoc

    elif len(mix_cont) == 1 and len(red_cont) == 1 and len(blue_cont) == 1:
        red_hull = cv2.convexHull(red_cont[0], False)
        img = cv2.drawContours(img, [red_hull], -1, (0, 0, 255), 1)

        blue_hull = cv2.convexHull(blue_cont[0], False)
        img = cv2.drawContours(img, [blue_hull], -1, (255, 0, 0), 1)

        if not(mousePress):
            mouse.press(Button.left)
            mousePress = True

        M_red = cv2.moments(red_hull)
        flag1 = False
        if M_red["m00"] != 0:
            flag1 = True
            c1X = int(M_red["m10"] / M_red["m00"])
            c1Y = int(M_red["m01"] / M_red["m00"])
            img = cv2.circle(img, (c1X, c1Y), 3, (0, 0, 255), -1)

        M_blue = cv2.moments(blue_hull)
        flag2 = False
        if M_blue["m00"] != 0:
            flag2 = True
            c2X = int(M_blue["m10"] / M_blue["m00"])
            c2Y = int(M_blue["m01"] / M_blue["m00"])
            img = cv2.circle(img, (c2X, c2Y), 3, (255, 0, 0), -1)

        if flag1 and flag2:
            cX = int((c1X+c2X)/2)
            cY = int((c1Y+c2Y)/2)
            mouseLoc = (sx - (sx * (cX/imw)), sy*(cY/imh))
            mouse.position = mouseLoc
    cv2.imshow("img", img)
    cv2.imshow("mix", mix_blur)
    cv2.imshow("red", red_blur)
    cv2.imshow("blue", blue_blur)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
