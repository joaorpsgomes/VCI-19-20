import numpy as np
import cv2
import time
#from matplotlib import pyplot as pltq
import argparse

def resize(image,scl):
    w_image = int(image.shape[1] * scl / 100)
    h_image = int(image.shape[0] * scl / 100)
    dim_image = (w_image,h_image)
    image_resized = cv2.resize(image, dim_image, interpolation = cv2.INTER_AREA)
    return image_resized

def estimateFrameRate(num_frame=30):
	cap = cv2.VideoCapture(0)
	
	start =time.time()

	for i in range(num_frame):
		_,frame =cap.read()

	stop = time.time()

	print("Capture {0:4d} frames in {1:3.2f} seconds   => frame rate = {2:3.2f}".format(num_frame,stop-start,num_frame/(stop-start)))


def saveimage():
	
	cv2.imwrite('./image.png',PlayVideoRealTime())
			

#A função PlayVideoRealTime() serve para mostrar a cãmara (do pc ou associada) em tempo real	

def PlayVideoRealTime():
	cap = cv2.VideoCapture(0)

	while(True):

		ret, frame= cap.read()

		cv2.imshow('frame',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

	return frame

def watermark():
	cap =cv2.VideoCapture(0)

	window_name="Watermark"
	font=cv2.FONT_HERSHEY_SIMPLEX
	org=(50,50)
	fontScale=1
	color=(0,0,0)
	thickness=2
	opacity=0.3
	

	while(True):

		ret, frame=cap.read()
		output=frame.copy()
		cv2.imshow('real',frame)

		watermark = cv2.putText(frame, 'VCI Group: 3', org, font, fontScale, color, thickness, cv2.LINE_AA)

		cv2.addWeighted(watermark, opacity, output, 1 - opacity,0, output)

		cv2.imshow(window_name,output)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.imwrite('./watermark.png',watermark)
			break

'''
	Add a image watermark in the current video that is captured by camera	
'''
def watermarkimage():
	cap = cv2.VideoCapture(0)										# capture images from camera
	image = cv2.imread('group_logo.jpg',1)		
	oH,oW = image.shape[:2]
    #image = np.dstack([image, np.ones((oH,oW), dtype="uint8") * 255])
	image_resized=resize(image,7)									# resize image
	watermark = cv2.cvtColor(image_resized, cv2.COLOR_BGR2BGRA)		# convert image BRG to GRAY	
	
	while(True):

		ret, frame= cap.read()	

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)				# convert image BRG to GRAY	
		frame_h, frame_w, frame_c = frame.shape

		overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
		watermark_h, watermark_w, watermark_c = watermark.shape

		for i in range(0, watermark_h):								# add the image in the wanted position
			for j in range(0, watermark_w):
				if watermark[i,j][3] != 0:
					offset = 10
					h_offset = frame_h - watermark_h - offset
					w_offset = frame_w - watermark_w - offset
					overlay[h_offset + i, w_offset+ j] = watermark[i,j]

		cv2.addWeighted(overlay, 0.50, frame, 0.9, 0, frame) 		# addWeighted(logo,intensity,frame,exposure,...) 

		frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)				# convert image BRA to BRG

		cv2.imshow('watermark',frame)				

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

def brigthness(alpha,beta):
	cap = cv2.VideoCapture(0)

	while(True):
		ret, frame =cap.read()
		
		result = cv2.addWeighted(frame, alpha ,np.zeros(frame.dtype),0,beta)	
		cv2.imshow('frame',frame)
		cv2.imshow('result',result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()

 
def white_balance():

	cap = cv2.VideoCapture(0)

	while(True):

		ret, frame= cap.read()

		cv2.imshow('frame',frame)

		result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
		avg_a = np.average(result[:, :, 1])
		avg_b = np.average(result[:, :, 2])
		result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
		result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
		result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

		cv2.imshow("white intensity",result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


'''
	Normlize the frame of video
'''
def intensitynormalization():

	cap=cv2.VideoCapture(0)

	while(True):
		ret, frame =cap.read()

		cv2.imshow('frame',frame)

		normalizedImg = np.zeros((800, 800))
		normalizedImg = cv2.normalize(frame,  normalizedImg, 0, 255, cv2.NORM_MINMAX)		# normalize the frame 
		cv2.imshow('dst_rt', normalizedImg)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

# Esta função, como o próprio nome diz, dá uso às funções que gravam e guardam os vídeos num formato, fps e dimensões pré-definidas

def SaveVideo():
	cap = cv2.VideoCapture(0)
	#fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
	fourcc = cv2.VideoWriter_fourcc(*'H264')        
	frames_per_second=20

	out = cv2.VideoWriter('output1.mp4', fourcc, frames_per_second, (640,  480))
	            
	while(cap.isOpened()):
	    ret, frame = cap.read()
	    
	    if ret==True:
	        frame = cv2.flip(frame,1)       # flip frame
	        out.write(frame)    # write the flipped frame
	        if cv2.waitKey(20) & 0xFF == ord('q'):
	            break
	    else:
	        break

	cap.release()
	out.release()
	cv2.destroyAllWindows()





parser = argparse.ArgumentParser()

parser.add_argument("-e","--FrameRate", help="print frame rate",
                    action="store_true")
parser.add_argument("-s","--SaveImage", help="save an image when 'q' is pressed",
                    action="store_true")

parser.add_argument("-p","--PlayVideo", help="Play Video real time",
                    action="store_true")

parser.add_argument("-w","--WaterMark", help="Play Video real time with water mark",
                    action="store_true")

parser.add_argument("-b","--brigthness", help="Play Video real time with brigthness enhanced",required=False, default=2)

parser.add_argument("-c","--contrast", help="Play Video real time with brigthness enhanced",required=False, default=30)


parser.add_argument("-t","--whitebalance", help="Play Video real time with White balance",
                    action="store_true")

parser.add_argument("-i","--Normalization", help="Play Video real time with Intensity Normalized",
                    action="store_true")

parser.add_argument("-v","--SaveVideo", help="Save Video",
                    action="store_true")

parser.add_argument("-m","--watermarkImage", help="Show Video with an image as a watermark",
                    action="store_true")

args = parser.parse_args()


if args.FrameRate:
	estimateFrameRate()
elif args.SaveImage:
	saveimage()
elif args.PlayVideo:
	PlayVideoRealTime()
elif args.WaterMark:
	watermark()
elif args.SaveVideo:
	SaveVideo()
elif args.whitebalance:
	white_balance()
elif args.Normalization:
	intensitynormalization()
elif args.watermarkImage:
	watermarkimage()	
elif args.brigthness:
	brigthness(float(args.brigthness), int(args.contrast))


