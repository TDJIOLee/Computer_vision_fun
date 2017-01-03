# USAGE
# python detection.py --face cascades/haarcascade_frontalface_default.xml --image images/img01.png

# import the necessary packages
from library.facedetector import FaceDetector
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
	help = "path to where the face cascade resides")
ap.add_argument("-i", "--image", required = True,
	help = "path to where the image file resides")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find faces in the image
fd = FaceDetector(args["face"])
faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
	minSize = (30, 30))
print "I found %d face(s)" % (len(faceRects))

# find eyes in the face
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# draw bonding box around an eye
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = image[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

# show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
