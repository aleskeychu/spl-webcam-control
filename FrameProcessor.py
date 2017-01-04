import sys
import numpy as np
import cv2


class FrameProcessor:

	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			raise Exception("Can't access web-camera")


	def get_next_frame(self):
		_, self.frame = self.cap.read()

	def threshold(self):
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR3GRAY)
		blurred = cv2.GaussianBlur(gray, (23, 23), 0)