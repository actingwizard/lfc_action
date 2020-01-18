#Import libraries
import cv2
import os
import numpy as np
import argparse
import imutils
from numpy import diff
import copy
import time
import matplotlib.pyplot as plt
from imageai.Prediction.Custom import CustomImagePrediction
from imageai.Detection.Custom import CustomObjectDetection

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from dbscan_test import *
from kmeans_model import *
from sklearn.cluster import KMeans
import colorsys

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
        #read image
        img = self.IMAGE
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        #save image after operations
        self.IMAGE = img
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        #save labels
        self.LABELS = kmeans.labels_
        #returning after converting to integer from float
        return self.COLORS.astype(int)

execution_path = os.getcwd()
prediction = CustomImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath("model_ex-100_acc-0.829508.h5")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=11)

map_to_classes= { 
    'team1left': '0', 
   	'team1down': '1', 
	'team1up': '2', 
    'team1right': '3',  
    'team2left': '4', 
   	'team2down': '5', 
	'team2up': '6', 
    'team2right': '7',     	
    'Team1GoalAttack': '8', 
   	'Team2GoalAttack': '9',
   	'team1': 1,
   	'team2': 2
}
#Reading the video
vidcap = cv2.VideoCapture('matchForLSTM.mp4')
#vidcap = cv2.VideoCapture('passes_video/pass10.mp4')
success,image = vidcap.read()
count = 1
success = True
idx = 0
coordinatesPreviousFrame = ''
scene = 'Midfield'
velocityArray = np.array([[0, 0, 0]])
ballTracking = 0
ballData = np.array([[0, 0, 0]])
coordinatesOfPlayersArray = []
oldData = []
colors = ('b','g','r')

#white range
lower_white = np.array([0,200,0])
upper_white = np.array([255,255,255])
#green range
lower_green = np.array([35,40, 20])
upper_green = np.array([80, 255, 255])

team1_color = np.array([0, 0, 0])
team2_color = np.array([0, 0, 0])
k_means_complete = 0

def append_to_file(filename, arr):
	f = open(filename, 'ab')
	np.savetxt(f, arr)
	f.close()

def process_countour(color_type, lower_range, upper_range,x,y,w,h):
	img = image[y:y+h,x:x+w]
	if color_type == "hsl":
		color_type = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	else:
		color_type = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(color_type, lower_range, upper_range)
	res1 = cv2.bitwise_and(img, img, mask=mask1)
	res1 = cv2.cvtColor(res1,cv2.COLOR_HLS2BGR)
	res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
	return cv2.countNonZero(res1)

def find_closest_player(coordinates):
	minDistance = 10000000
	closestPlayer= ''
	closestPlayerTeam = 'team1'
	if ('ball' in coordinates):
		ballX = coordinates['ball'][0]['x1']
		ballY = coordinates['ball'][0]['y1']
		for teamNumber in coordinates:
			if (teamNumber == 'team1') or (teamNumber == 'team2'):
				for playerIndex in coordinates[teamNumber]:
					playerX = coordinates[teamNumber][playerIndex]['x2']
					playerY = coordinates[teamNumber][playerIndex]['y2']
					if ((playerX-ballX)**2+(playerY-ballY)**2 < minDistance):
						minDistance = (playerX-ballX)**2+(playerY-ballY)**2
						closestPlayer = coordinates[teamNumber][playerIndex]
						closestPlayerTeam = teamNumber
	return closestPlayerTeam

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-112--loss-0006.162.h5") 
detector.setJsonPath("detection_config.json")
detector.loadModel()
        
#Read the video frame by frame
while success:

	success,image = vidcap.read()
	count += 1
	print(image.shape)
	coordinates = {}
	number_team1_players = 0
	number_team2_players = 0
	number_ball = 0
	action = ''
	cv2.imwrite('image.jpg',image)
	if (count % 15 == 0):
		
		start_time = time.time()
		predictions, probabilities = prediction.predictImage("image.jpg", result_count=1)
		print("scene_recognition_time: " + str(time.time() - start_time))
		scene = predictions[0]

	time_detectin_players_start = time.time()
	#converting into hsv image
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	#Define a mask ranging from lower to uppper
	mask = cv2.inRange(hsv, lower_green, upper_green)
	#Do masking
	maskInv = cv2.bitwise_not(mask)
	resultHSV = cv2.bitwise_and(hsv, hsv, mask=maskInv)
	result = cv2.cvtColor(resultHSV, cv2.COLOR_HSV2BGR)

	res = cv2.bitwise_and(image, image, mask=mask)
	#convert to hsv to gray
	res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
	res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	#Defining a kernel to do morphological operation in threshold image to 
	#get better output.
	kernel = np.ones((9,9),np.uint8)
	thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	borderMask = cv2.bitwise_not(thresh)
	kernel = np.ones((22, 22),np.uint8)
	borderMask = cv2.dilate(borderMask,kernel,iterations = 1)

	
		
	if (scene == 'Midfield' or scene == "GoalAttack" or scene == "Corner" or scene == 'CounterAttack'):
		checker = 0
		prev = 0
		font = cv2.FONT_HERSHEY_SIMPLEX
		#time_only_players = time.time()
		#find contours in threshold image     
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		numberOfCountoursOfPlayers = 0
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)

			#Detect players
			if(h>=1.5*w and h<4*w and w>4 and h>=6):
				time_histogram = time.time()
				numberOfCountoursOfPlayers += 1
				mask = np.zeros((result.shape[0], result.shape[1]), dtype=np.uint8)
				mask[y : y + h, x : x + w] = 1
				i_hist = np.array([])
				for ix, cl in enumerate(colors):
					#time_histogram = time.time()
					hist = cv2.calcHist([result], [ix], mask, [255], [0, 255])
					#print("time_histogram: " + str(time.time()-time_histogram))
					hist = np.delete(hist, 0)
					i_hist = np.append(i_hist, np.argmax(hist))	
				#print("time_histogram: " + str(time.time()-time_histogram))
				

				#time_not_histogram = time.time()
				distance1=((team1_color[0]-i_hist[0])**2+(team1_color[1]-i_hist[1])**2+(team1_color[1]-i_hist[1])**2)	
				distance2=((team2_color[0]-i_hist[0])**2+(team2_color[1]-i_hist[1])**2+(team2_color[1]-i_hist[1])**2)


				if(distance1<distance2):
					# cv2.putText(image, 'team1', (x-2, y-2), font, 0.8, (int(team1_color[2]), int(team1_color[1]), int(team1_color[0])), 2, cv2.LINE_AA)
					# cv2.rectangle(image,(x,y),(x+w,y+h),(int(team1_color[2]), int(team1_color[1]), int(team1_color[0])),3)
					if 'team1' in coordinates:
						coordinates['team1'].update({number_team2_players: {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h }})
					else:
						coordinates.update({'team1': {number_team2_players: {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h }}})
					number_team2_players += 1
				elif(distance1>distance2):
					#Mark red jersy players as RMA
					# cv2.putText(image, 'team2', (x-2, y-2), font, 0.8, (int(team2_color[2]), int(team2_color[1]), int(team2_color[0])), 2, cv2.LINE_AA)
					# cv2.rectangle(image,(x,y),(x+w,y+h),(int(team2_color[2]), int(team2_color[1]), int(team2_color[0])),3)
					if 'team2' in coordinates:
						coordinates['team2'].update({number_team1_players: {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h }})
					else:
						coordinates.update({'team2': {number_team1_players: {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h }}})
					number_team1_players += 1
				else:
					pass
				#print("time_not_histogram: " + str(time.time()-time_not_histogram))
		#print("time_only_players: " + str(time.time()-time_only_players))
		if numberOfCountoursOfPlayers>=16 and k_means_complete == 0:
			image_for_k_means = result.copy()
			
			cv2.imshow('result', result)
			print("Starting K-means...")
			for i in range(720):
				for j in range(1280):
					if borderMask[i][j] == 0 or mask[i][j] == 255:
						image_for_k_means[i][j] = 0
			cv2.imshow('image_for_k_means', image_for_k_means)
			clusters = 4
			start_time = time.time()
			dc = DominantColors(image_for_k_means, clusters) 
			dominant_colors = dc.dominantColors()
			print("dominant_colors_finding_time: " + str(time.time() - start_time))
			team1_color = np.array([dominant_colors[2][2], dominant_colors[2][1], dominant_colors[2][0]])
			team2_color = np.array([dominant_colors[3][2], dominant_colors[3][1], dominant_colors[3][0]])
			k_means_complete = 1
			print("Completing K-means...")
			continue

				
		start_time = time.time()
		detections = detector.detectObjectsFromImage(input_image="image.jpg", output_image_path="result.jpg")
		print (time.time() - start_time)
		print("detections: " + str(detections))
		for detection in detections:
			print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
			x1 = detection['box_points'][0]
			y1 = detection['box_points'][1]
			x2 = detection['box_points'][2]
			y2 = detection['box_points'][3]
			cv2.putText(image, 'ball', (x1-2, y1-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
			cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
			if 'ball' in coordinates:
				coordinates['ball'].update({number_ball: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 }})
			else:
				coordinates.update({'ball': {number_ball: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 }}})
			number_ball = 0 
			velocityArray = np.append(velocityArray, [[count, x1, y1]], axis=0)
			break


		
			
		contours,hierarchy = cv2.findContours(borderMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		# for c in contours:
		# 	x,y,w,h = cv2.boundingRect(c)
		# 	if (w>500):
		# 		nzCount4 = process_countour("hsv",lower_green, upper_green, x, y, w, h)
		# 		if(nzCount4>=20):
		# 			#Mark borders
		# 			# cv2.drawContours(image, [c], 0, (0, 0, 255), 4)
		# 			# cv2.putText(image, 'borders', (x-2, y-2), font, 2, (0,0,255), 4, cv2.LINE_AA)
		# 		else:
		# 			pass

		coordinates.update({'frameNumber' : count})
		closestPlayerTeam = find_closest_player(coordinates)
		coordinatesOfPlayersArray = []
		coordinatesOfPlayersArray.append(count)
		coordinatesOfPlayersArray.append(map_to_classes[closestPlayerTeam])
		coordinatesOfPlayersArray.append(number_team1_players)

	
		i = 0
		if 'team1' in coordinates:
			for player in coordinates['team1']:
				coordinatesOfPlayersArray.append(coordinates['team1'][player]['x2'])
				coordinatesOfPlayersArray.append(coordinates['team1'][player]['y2'])
				i = i + 1
				if i==11:
					break
		for x in range (i, 11):
			coordinatesOfPlayersArray.append(0)
			coordinatesOfPlayersArray.append(0)
		coordinatesOfPlayersArray.append(number_team2_players)
		
		i = 0
		if 'team2' in coordinates:
			for player in coordinates['team2']:
				coordinatesOfPlayersArray.append(coordinates['team2'][player]['x2'])
				coordinatesOfPlayersArray.append(coordinates['team2'][player]['y2'])
				i = i + 1
				if i==11:
					break
		for x in range (i, 11):
			coordinatesOfPlayersArray.append(0)
			coordinatesOfPlayersArray.append(0)
		
		oldData.append(coordinatesOfPlayersArray)
		print("frame: " + str(count))
		print("Player_detection_time: " + str(time.time()-time_detectin_players_start))


		if 'ball' in coordinates:
			coordinatesPreviousFrame = copy.deepcopy(coordinates)	
		
		if 'ball' in coordinates:
			ballTracking  = 0
		else:
			ballTracking  = ballTracking + 1
		if ballTracking == 30 or success == False:
			time_ball_noice_removal = time.time()
			X = prep_data(velocityArray)	
			ranges = get_pass_ranges(X)
			true_cdn, true_ranges_idx = true_cdn_ball(X, ranges)
			print("ball_noice_removal_time: " + str(time.time() - time_ball_noice_removal))

			db = np.array(oldData).astype(int)
			writing_list = []
		
			# NOTE: fn_idx is the index of one after finish
			for (st_idx, fn_idx) in true_ranges_idx:
				# 1 = short pass 8-20
				# 2 = long pass 20-50
				# 3 = traversal
				action = None
				s_idx = -1
				f_idx = -1
				for fm, x, y, cdn, grp in X:
					if fm >= X[st_idx, 0] and (np.where(true_cdn[:, 0] == fm)[0]).size != 0 and s_idx == -1:
						s_idx = np.where(true_cdn[:, 0] == fm)[0][0]

					if fm <= X[fn_idx - 1, 0] and np.where(true_cdn[:, 0] == fm)[0].size != 0:
						f_idx = np.where(true_cdn[:, 0] == fm)[0][0]

				ball_st_x = true_cdn[s_idx, 1]
				ball_st_y = true_cdn[s_idx, 2]
				ball_fn_x = true_cdn[f_idx, 1]
				ball_fn_y = true_cdn[f_idx, 2]					

				diff_x = ball_fn_x - ball_st_x
				diff_y = ball_fn_y - ball_st_y

				if (diff_x >= 0 and diff_x >= abs(diff_y)):
					ratio_xy = diff_y / diff_x
					if ratio_xy > 0.41 and ratio_xy <= 1:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 1
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 2
						else:
							action = 3

					elif ratio_xy <= 0.41 and ratio_xy > 0:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 4
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 5
						else:
							action = 6
					
					elif ratio_xy <= 0 and ratio_xy > -0.41:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 7
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 8
						else:
							action = 9
					
					if ratio_xy <= -0.41 and ratio_xy > -1:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 10
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 11
						else:
							action = 12
					

				if (diff_y <= 0 and abs(diff_y) >= abs(diff_x)):
					if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
						action = 13
					elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
						action = 14
					else:
						action = 15				

				if (diff_x <= 0 and abs(diff_x) >= abs(diff_y)):
					ratio_xy = diff_y / diff_x
					if ratio_xy > 0.41 and ratio_xy <= 1:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 16
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 17
						else:
							action = 18

					elif ratio_xy <= 0.41 and ratio_xy > 0:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 19
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 20
						else:
							action = 21
					
					elif ratio_xy <= 0 and ratio_xy > -0.41:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 22
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 23
						else:
							action = 24
					
					if ratio_xy <= -0.41 and ratio_xy > -1:
						if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
							action = 25
						elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
							action = 26
						else:
							action = 27


				if (diff_y >= 0 and diff_y >= abs(diff_x)):
					if X[fn_idx - 1, 0] - X[st_idx, 0] >= 8 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 15:
						action = 28
					elif X[fn_idx - 1, 0] - X[st_idx, 0] >= 16 and X[fn_idx - 1, 0] - X[st_idx, 0] <= 40:
						action = 29
					else:
						action = 30


				for idx in np.arange(st_idx, fn_idx):
					if (np.where(true_cdn[:, 0] == X[idx, 0])[0]).size != 0:
						tmp_list = []
						team_controlling_ball = db[np.where(db[:, 0] == X[idx, 0])[0][0], 1]
						ball_x = true_cdn[np.where(true_cdn[:, 0] == X[idx, 0])[0][0], 1]
						ball_y = true_cdn[np.where(true_cdn[:, 0] == X[idx, 0])[0][0], 2]
						num_team1 = db[np.where(db[:, 0] == X[idx, 0])[0][0], 2]
						num_team2 = db[np.where(db[:, 0] == X[idx, 0])[0][0], 25]

						tmp_list.append(X[idx, 0]) # frame
						tmp_list.append(action)    # action_label

						tmp_list.append(team_controlling_ball) # team controlling ball
						tmp_list.append(ball_x) # ball x coord
						tmp_list.append(ball_y) # ball y coord
						tmp_list.append(num_team1)  # of players in team1
						tmp_list.append(num_team2)  # of players in team2

						team1_xy = db[np.where(db[:, 0] == X[idx, 0])[0][0], 3:25]
						for m in team1_xy:
							tmp_list.append(m)

						team2_xy = db[np.where(db[:, 0] == X[idx, 0])[0][0], 26:48]
						for m in team2_xy:
							tmp_list.append(m)
						
						writing_list.append(tmp_list)

			time_dataset_appending = time.time()
			append_to_file('dataset_30_actions_fcb_rma_full.csv', writing_list)
			# print("dataset_appending_time: " + str(time.time() - time_dataset_appending))
			print ('saving..')
		
			oldData = []
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			velocityArray = np.zeros((1, 3))

	# count += 1
	# cv2.imshow('Match Detection',image)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
	# success,image = vidcap.read()
	cv2.imshow('Match Detection',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

oldData = np.array(oldData).astype(int)
np.set_printoptions(threshold=np.inf)
np.savetxt("foo.csv", velocityArray, delimiter=",")
vidcap.release()
cv2.destroyAllWindows()
 
