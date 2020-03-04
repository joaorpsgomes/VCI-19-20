import numpy as np
import cv2


def changeColorSpace(colorspace, frame):
	if colorspace=='hsv':
		return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	if colorspace=='yuv':
		return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)



cap = cv2.VideoCapture(0)

while(True):

	ret, frame= cap.read()

	cv2.imshow('frame',frame)
	frame_2=changeColorSpace('hsv',frame)
	cv2.imshow('hsv',frame_2)
	frame_3=changeColorSpace('yuv',frame)
	cv2.imshow('yuv',frame_3)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

