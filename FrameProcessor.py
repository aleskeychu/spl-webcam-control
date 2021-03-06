import sys
import numpy as np
import pyautogui as pag
import cv2
import argparse
import math

from functools import reduce
from collections import deque, namedtuple

pag.FAILSAFE = False
MAX_ANGLE = 80
SAMPLES = 5
HLS = namedtuple("HLS", "h, l, s")
bounds = HLS(10, 20, 40)
Point = namedtuple("Point", "x, y")

class ROI:

	def __init__(self, x=0, y=0, height=12, width=12, thickness=2, color=(0, 255, 0)):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.thickness = thickness
		self.color = color

	def draw(self, array):
		cv2.rectangle(array, (self.x, self.y), (self.x+self.width, self.y + self.height), self.color, self.thickness)


class FrameProcessor:

	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			raise Exception("Can't access web-camera")
		Size = namedtuple('Size', 'width height')
		self.screen_size = Size(*pag.size())
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_size[1])
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_size[0])
		self.hand_centers = deque(maxlen=15) # collectionq for storing last 100 hand positions, tuples of format (x, y)
		self.fingers = deque(maxlen=10) # collection for storing very approximate number of fingers
		self.recent_click = False # trace whether clicks were made recently	
		self._positions = ((0.4, 0.5), (0.2, 0.6), (0.35, 0.65), (0.35, 0.55), (0.4, 0.65), (0.4, 0.55), (0.5, 0.65), (0.5, 0.6), (0.5, 0.55), (0.37, 0.6), (0.26, 0.6)) # positions to draw boxes at
		self.rois = None
		self.samples = [] # color samples from each roi
		

	def get_next_frame(self):
		_, self.frame = self.cap.read()
		self.frame = cv2.flip(self.frame, 1)


	def get_color_samples(self):		
		while True:	
			self.get_next_frame()
			fr = self.frame.copy()
			if not self.rois:
				self.rois = [ROI(int(fr.shape[1] * i[1]), int(fr.shape[0]* i[0])) for i in self._positions] # initiliaze boxes from which to get color sampels
			for roi in self.rois:
				roi.draw(fr)
			cv2.imshow('lol', fr)
			if cv2.waitKey(1) & 0xFF == ord('d'):
				hls = cv2.cvtColor(fr, cv2.COLOR_BGR2HLS)
				colors = []
				for roi in self.rois:
					h, l, s = [[] for i in range(3)]
					subroi = hls[roi.y+roi.thickness : roi.y+roi.height-roi.thickness, roi.x+roi.thickness : roi.x+roi.width-roi.thickness] # area inside box
					for row in subroi:
						for pixel in row:
							h.append(pixel[0])
							l.append(pixel[1])
							s.append(pixel[2])
					h.sort()
					l.sort()
					s.sort()
					length = len(h)
					self.samples.append(HLS(h[length // 2], l[length // 2], s[length // 2])) # array of tuples of median values
				break	
				

	def get_bounds(self):
		Bound = namedtuple("Bound", "bot top")
		self.bounds = []
		for sample in self.samples:
			h = Bound(bounds.h if sample.h - bounds.h >= 0 else sample.h, bounds.h if sample.h + bounds.h <= 255 else 255 - sample.h)
			l = Bound(bounds.l if sample.l - bounds.l >= 0 else sample.l, bounds.l if sample.l + bounds.l <= 255 else 255 - sample.l)
			s = Bound(bounds.s if sample.s - bounds.s >= 0 else sample.s, bounds.s if sample.s + bounds.s <= 255 else 255 - sample.s)
			lower = np.array([sample.h - h.bot, sample.l - l.bot, sample.s - s.bot])
			upper = np.array([sample.h + h.top, sample.l + l.top, sample.s + s.top])
			self.bounds.append((lower, upper))


	def get_threshold_hsv(self):
		hls = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS)
		thresholded_frames = []
		for bound in self.bounds:
			fr = cv2.inRange(hls.copy(), bound[0], bound[1])
			thresholded_frames.append(fr)
		self.thr = reduce(np.bitwise_or, thresholded_frames)
		self.threshold_from_hls = cv2.medianBlur(self.thr, 7)


	def erode_and_dilate(self):
		mask = self.threshold_from_hls
		kernel_square = np.ones((9, 9),np.uint8)
		kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		dilation = cv2.dilate(mask,kernel_ellipse)
		erosion = cv2.erode(dilation,kernel_square)    
		dilation2 = cv2.dilate(erosion,kernel_ellipse)    
		filtered = cv2.medianBlur(dilation2,5)
		kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
		dilation2 = cv2.dilate(filtered,kernel_ellipse)
		kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		dilation3 = cv2.dilate(filtered,kernel_ellipse)
		self.median = cv2.medianBlur(dilation2,5)
	
	def get_threshold(self):
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (23, 23), 0) #blurring image to smooth acute borders to ease thresholding
		_, self.threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Work on the assumption that the hand is closer to the camera and so it occupies the biggest area of the screen.
	def get_hand_contour(self, thr):
		_, self.contours, __ = cv2.findContours(thr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
		self.hull_points = np.array(self.hull_points, dtype=np.int16)

	@staticmethod
	def angle(a, b, c):
		ab = Point(b.x - a.x, b.y - a.y)
		cb = Point(b.x - c.x, b.y - c.y)
		dotprod = ab.x * cb.x + ab.y * cb.y
		mod_ab = ab.x ** 2 + ab.y ** 2
		mod_cb = cb.x ** 2 + cb.y ** 2
		cos = dotprod / math.sqrt(mod_ab * mod_cb)
		acos = math.acos(cos)
		degrees = acos / math.pi * 180
		return degrees

	def get_defects(self):
		defects = cv2.convexityDefects(self.hand_contour, self.hull_contour) 
		self.all_defects = defects
		x, y, w, h = cv2.boundingRect(self.hand_contour)
		min_convex_hull_length = 0.03 * (w + h)

		self.defects = []
		for defect in defects:
			start, end, farthest_index, distance = defect[0]
			distance = distance / 256 # distance contains 8 fractional bits, gotta get rid of them
			print("{} = {}".format(min_convex_hull_length, distance))
			if distance < min_convex_hull_length:
				print("missed bc ditance")
				continue
			a = Point(self.hand_contour[start][0][0], self.hand_contour[start][0][1])
			b = Point(self.hand_contour[farthest_index][0][0], self.hand_contour[farthest_index][0][1])
			c = Point(self.hand_contour[end][0][0], self.hand_contour[end][0][1])
			if FrameProcessor.angle(a, b, c) > MAX_ANGLE:
				print("missed bc angle")
				continue
			self.defects.append(defect)


	def get_center(self):
		moments = cv2.moments(self.hand_contour)
		mom_x = int(moments['m10'] / moments['m00'])
		mom_y = int(moments['m01'] / moments['m00'])
		self.hand_centers.append((mom_x, mom_y))


	def get_palm(self):
		cont = self.hand_contour.copy()
		defects = self.defects.copy()


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
		if delta <= -3 and not self.recent_click: # 3 is an arbitrary, empiricaly found number of fingers
			self.recent_click = True
			pag.click()
			# print('click Done')
		elif delta >= 3 and self.recent_click:
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
		fp.get_color_samples()
		fp.get_bounds()
		cv2.destroyAllWindows()
		while True:
			fp.get_next_frame()
			fp.get_threshold_hsv()
			fp.erode_and_dilate()
			fp.get_hand_contour(fp.median.copy())
			fp.get_hulls()
			fp.get_defects()
			print(fp.contours)
			print(fp.defects)
			# sys.exit(0)
			print(fp.hand_contour)
			for defect in fp.all_defects:
				cv2.circle(fp.frame, tuple(fp.hand_contour[defect[0][0]][0]), 5, (255, 0, 0), 4)
				cv2.circle(fp.frame, tuple(fp.hand_contour[defect[0][1]][0]), 5, (255, 0, 0), 4)
				cv2.circle(fp.frame, tuple(fp.hand_contour[defect[0][2]][0]), 5, (255, 0, 255), 4)
			cv2.drawContours(fp.frame, fp.hand_contour, -1, (0, 255, 0), 3)
			cv2.imshow('ead', fp.median)
			cv2.imshow('fr', fp.frame)
			cv2.waitKey(1)
		if args.show_video:
			img = np.zeros(fp.frame.shape)
			cv2.drawContours(img, fp.hand_contour, -1, (0, 255, 0), 3)
			cv2.circle(img, fp.hand_centers[-1], 10, (255, 0, 0))
			cv2.imshow('contours', img)
			cv2.imshow('threshold', fp.threshold)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()	