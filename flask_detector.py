#!/usr/bin/env python3

# import the necessary packages
from multiprocessing.pool import INIT
from flask import Flask, jsonify, abort, make_response
from flask_restful import Api, Resource, reqparse, fields, marshal
import threading 
import numpy as np
from urllib.error import URLError, HTTPError
import cv2
import time
import requests
import sys
from detect_openvino_flask import load_model, inference

status = [
    {
        'human_detected': 'no'
    }
]

status_fields = {
	'human_detected': fields.String
}

INIT_TIME = 2
BASE_PATH = "C:/Users/arite/Desktop/KMBIC/Collins Aerospace final codes/499-people-v1/overhead_person_detector/"
IMG_PATH = BASE_PATH + "flask_inference.jpg"
# THRESHOLD = 190
# TM_THRESHOLD = 0.8
# THRESHOLD_MATCH = 0.92
# THRESHOLD_MATCH_LIFT = 70

class occupancyDetectionAPI(Resource):
	def __init__(self):
		self.reqparse = reqparse.RequestParser()
		self.reqparse.add_argument('occupancy', type=str, location='json')
		super(occupancyDetectionAPI, self).__init__()

	def get(self):
		global status
		return {'status': marshal(status[0], status_fields)}


class occupancyRestServer(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		threading.Thread.daemon = True
	
	def run(self):
			errors = {
				'ResourceDoesNotExist': {
				'message': "A resource with that ID no longer exists.",
				'status': 410,
				'extra': "Any extra information you want.",
				},
			}

			app = Flask(__name__, static_url_path="")
			api = Api(app, errors=errors)
			api.add_resource(occupancyDetectionAPI, '/api/v1.0/occupancy')
			app.run(host='0.0.0.0')
			#app.run()


class occupancyDetection(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		threading.Thread.daemon = True
		self.mobotix_url = "http://192.168.0.201/record/current.jpg?size=960X960"
		self.iter = 0

	#OpenCV, NumPy, and urllib
	def url_to_image(self, url):
		# download the image, convert it to a NumPy array, and then read
		# it into OpenCV format
		try:	
			resp = requests.get(self.mobotix_url)
			image = np.asarray(bytearray(resp.content), dtype="uint8")
			image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		except:
			print("Extract image error.")
		return image

	def run(self):
		global status
		OVExec = load_model(
			model_xml_path= BASE_PATH + "best_openvino_model/best.xml", 
			model_bin_path= BASE_PATH + "best_openvino_model/best.bin", 
			target_device= "CPU")

		while True:

			#ret, img = vid.read()
			resp = requests.get(self.mobotix_url)
			image = np.asarray(bytearray(resp.content), dtype="uint8")
			img = cv2.imdecode(image, cv2.IMREAD_COLOR)
			cropped_img = img[:, 160:1120]
			cv2.imwrite(IMG_PATH, cropped_img)



			if self.iter > INIT_TIME:	
				if inference(
					input_path= IMG_PATH,
					OVExec= OVExec,
					output_dir= BASE_PATH + "best_openvino_model/output", 
					threshold= 0.4,
					debug= True, 
					save = True):
					status[0]['human_detected'] = 'yes'
					print("Human detected")
				else:
					status[0]['human_detected'] = 'no'
					print("Human not detected")
			
			self.iter = self.iter + 1
			time.sleep(1)			


def start_runner():
    def start_loop():
        not_started = True
        while not_started:
            print('In start loop')
            try:
                r = requests.get('http://127.0.0.10:5000/api/v1.0/occupancy')
                if r.status_code == 200:
                    print('Server started, quitting start_loop')
                    not_started = False
                print(r.status_code)
            except:
                print('Server not yet started')
            time.sleep(2)

    print('Started runner')
    thread = threading.Thread(target=start_loop)
    thread.start()

def exitApp():
    sys.exit()

if __name__ == '__main__':

	start_runner()

	restServer = occupancyRestServer()
	restServer.start()

	occupancyStatus = occupancyDetection()
	occupancyStatus.start()

	restServer.join()
	occupancyStatus.join()