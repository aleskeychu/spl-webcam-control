import sys
import numpy as np
import pyautogui as pag
import cv2
import argparse
import math

from collections import deque

pag.FAILSAFE = False
MIN_CONVEX_HULL_LENGTH = 100

class FrameProcessor:


	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			raise Exception("Can't access web-camera")
		self.screen_size = pag.size()
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_size[1])
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_size[0])
		self.hand_centers = deque(maxlen=15) # collectionq for storing last 100 hand positions, tuples of format (x, y)
		self.fingers = deque(maxlen=10) # collection for storing very approximate number of fingers
		self.recent_click = False # trace whether clicks were made recently

	def get_next_frame(self):
		_, self.frame = self.cap.read()
		self.frame = cv2.flip(self.frame, 1)

	# couldn't properly implement, commented for later fixes
	# def erose_and_dilate(self):
	# 	self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
	# 	mask = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

	# 	dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
	# 	erosion = cv2.erode(dilation, kernel_square, iterations=1)    
	# 	dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)    
	# 	filtered = cv2.medianBlur(dilation2, 5)
	# 	kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
	# 	dilation2 = cv2.dilate(filtered, kernel_ellipse,iterations=1)
	# 	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	# 	dilation3 = cv2.dilate(filtered,kernel_ellipse, iterations=1)
	# 	median = cv2.medianBlur(dilation2, 5)
	# 	_, self.threshold2 = cv2.threshold(median, 127, 255, 0)

	
	def get_threshold(self):
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (23, 23), 0) #blurring image to smooth acute borders to ease thresholding
		_, self.threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Work on the assumption that the hand is closer to the camera and so it occupies the biggest area of the screen.
	def get_hand_contour(self):
		_, self.contours, __ = cv2.findContours(self.threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		maxArea, cont = 0, None
		for c in self.contours:
			area = cv2.contourArea(c)
			if area > maxArea:
				maxArea = area
				cont = c
		self.hand_contour = cv2.approxPolyDP(cont, 0.001*cv2.arcLength(cont, True), True)


	def get_hand_dims(self):
		self.hand_x, self.hand_y, self,hand_width, self.hand_height = cv2.boundingRect(self.hand_contour)


	def get_hulls(self):
		self.hull_contour = cv2.convexHull(self.hand_contour, returnPoints=False) # returns indeces
		self.hull_points = [self.hand_contour[i[0]] for i in self.hull_contour] # actual points
		self.hull_points = np.array(self.hull_points, dtype=np.int32)


	def get_defects(self):
		self.defects = cv2.convexityDefects(self.hand_contour, self.hull_contour) 
		number_of_defects = 0
		for defect in self.defects:
			start, end, farthest_index, distance = defect[0]
			distance = distance / 256 # distance contains 8 fractional bits, gotta get rid of them
			if distance >= MIN_CONVEX_HULL_LENGTH:
				number_of_defects += 1
		self.fingers.append(number_of_defects)


	def get_center(self):
		moments = cv2.moments(self.hand_contour)
		mom_x = int(moments['m10'] / moments['m00'])
		mom_y = int(moments['m01'] / moments['m00'])
		self.hand_centers.append((mom_x, mom_y))


	def move(self):
		if len(self.hand_centers) < 2:
			return
		dots = self.hand_centers
		delta = ((dots[-1][0] - dots[-2][0]) * 2, (dots[-1][1] - dots[-2][1]) * 2) # calculatuing distance between 2 latest hand positions
		if math.sqrt(delta[0]**2 + delta[1]**2) < (self.screen_size[0] / 10): # filtering big distances as they are probably algorithm errors
			pag.moveRel(*delta)


	def register_click(self):
		delta = self.fingers[-1] - self.fingers[0]
		# print(delta)
		if delta < -3 and not self.recent_click: # 3 is an arbitrary number of fingers found empiricaly
			self.recent_click = True
			pag.click()
			# print('click Done')
		elif delta > 3 and self.recent_click:
			# print('click Undone')
			self.recent_click = False


	def run(self):
		fp.get_next_frame()
		fp.get_threshold()
		fp.get_hand_contour()
		fp.get_hulls()
		fp.get_center()
		fp.get_defects()
		fp.move()
		fp.register_click()




if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--show-video", help="also play video from the webcam with various filters", action="store_true")
	args = parser.parse_args()

	fp = FrameProcessor()
	while True:
		fp.run()
		if args.show_video:
			img = np.zeros((fp.frame.shape[0], fp.frame.shape[1] * 2), np.uint8)
			cv2.drawContours(img, fp.hand_contour, -1, (0, 255, 0), 3)
			cv2.circle(img, fp.hand_centers[-1], 10, (255, 0, 0))
			roi = img[:,fp.frame.shape[1]:]
			img[:,fp.frame.shape[1]:] = fp.threshold
			cv2.imshow('frame', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()