import cvlib as cv
import cv2
import numpy as np
from cvlib.object_detection import draw_bbox


def nothing(x):
    print(x)

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def object_detect():
    cap = cv2.VideoCapture('cambada_video.mp4')
    #cv2.namedWindow('image')

    while(True):
        ret, frame = cap.read()
        frame_ori=resize(frame,80)


        hsv = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2HSV)

        #tracking_ball()
        lower_hsv_ball = np.array([22, 77, 88])
        higher_hsv_ball = np.array([41, 254, 255])
        mask_ball = cv2.inRange(hsv, lower_hsv_ball, higher_hsv_ball)
        #tracking_blue_team()
        lower_hsv_blue = np.array([88, 90, 46])
        higher_hsv_blue = np.array([106, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_hsv_blue, higher_hsv_blue)
        #tracking_orange_team()
        lower_hsv_orange = np.array([0, 89, 0])
        higher_hsv_orange = np.array([20, 255, 196])
        mask_orange = cv2.inRange(hsv, lower_hsv_orange, higher_hsv_orange)

        #mask = mask_ball+mask_blue+mask_orange
        #im = cv2.bitwise_or(frame_ori, frame_ori, mask=mask)

        im = cv2.bitwise_and(frame_ori, frame_ori, mask=mask_ball)


        bbox, label, conf = cv.detect_common_objects(frame_ori)

        output_image = draw_bbox(frame_ori, bbox,label, conf)

        cv2.imshow('out',output_image)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
                break

    cv2.destroyAllWindows()