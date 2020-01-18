#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import time

ACTION_NUM = 30
MAX_SEQ = 0
#552, 31.22


# NOTE: ranges is inclusive
def search_ranges_idx(dataset):
	active_action, st, fn = (dataset[0, 1], 0, -1)
	ranges_idx = []
	for i in range(len(dataset)):
		if active_action != dataset[i, 1]:
			fn = i - 1
			ranges_idx.append((st, fn))
			st = i
			active_action = dataset[i, 1]
	ranges_idx.append((st, len(dataset) - 1))
	return ranges_idx

# return length of maximum 2 consecutive sequences
def find_max_seq(ranges_idx, action_seq_num = 2):
	max_seq = -1
	for i in range(len(ranges_idx) - action_seq_num + 1):
		seq_sum = 0
		for j in range(action_seq_num):
			seq = ranges_idx[i + j][1] - ranges_idx[i + j][0] + 1
			seq_sum += seq
		if seq_sum > max_seq:
			max_seq = seq_sum
	MAX_SEQ = max_seq
	return max_seq

def load_data_model(filename, action_seq_num = 2):
	df = pd.read_csv(filename, sep=' ', header=None)
	dataset = df.values.astype(int)
	dataset = dataset[:, :]
	ranges_idx = search_ranges_idx(dataset)
	max_seq = find_max_seq(ranges_idx, action_seq_num)
	modified_dataset_X, modified_dataset_y = (np.array([]), np.array([]))
	# print (np.array(ranges_idx))
	# for i in range(len(ranges_idx) - action_seq_num):
	if len(ranges_idx) > 1:
	# for i in range(1):
		i = len(ranges_idx) - action_seq_num
		inst_matrix = np.zeros((287, dataset.shape[1])).astype(int)
		s_ix = ranges_idx[i][0]
		f_ix = ranges_idx[i + action_seq_num - 1][1]
		inst_matrix[inst_matrix.shape[0] - dataset[s_ix:f_ix + 1].shape[0]:] = dataset[s_ix : f_ix + 1]
		if modified_dataset_X.size == 0:
			modified_dataset_X = inst_matrix.reshape(1, 287, dataset.shape[1])
			# modified_dataset_y = np.array([dataset[f_ix + 1, 1]]).reshape(-1, 1)
		else:
			modified_dataset_X = np.vstack((modified_dataset_X, inst_matrix.reshape(1, 287, dataset.shape[1])))
			# modified_dataset_y = np.vstack((modified_dataset_y, np.array([dataset[f_ix + 1, 1]]).reshape(-1, 1)))
	return modified_dataset_X

def load_data(filename, action_seq_num = 2):
	df = pd.read_csv(filename, sep=' ', header=None)
	# dataset = df.values.astype(int)
	dataset = df.values
	dataset = dataset[:, :]
	ranges_idx = search_ranges_idx(dataset)
	max_seq = find_max_seq(ranges_idx, action_seq_num)
	modified_dataset_X, modified_dataset_y = (np.array([]), np.array([]))
	# print ('dataset', dataset.shape)
	# print ('ranges_idx', ranges_idx)
	# print ('max_seq', max_seq)
	for i in range(len(ranges_idx) - action_seq_num):
		inst_matrix = np.zeros((max_seq, dataset.shape[1])).astype(int)
		s_ix, f_ix = (ranges_idx[i][0], ranges_idx[i + action_seq_num - 1][1])
		inst_matrix[inst_matrix.shape[0] - dataset[s_ix:f_ix + 1].shape[0]:] = dataset[s_ix : f_ix + 1]
		if modified_dataset_X.size == 0:
			modified_dataset_X = inst_matrix.reshape(1, max_seq, dataset.shape[1])
			modified_dataset_y = np.array([dataset[f_ix + 1, 1]]).reshape(-1, 1)
		else:
			modified_dataset_X = np.vstack((modified_dataset_X, inst_matrix.reshape(1, max_seq, dataset.shape[1])))
			modified_dataset_y = np.vstack((modified_dataset_y, np.array([dataset[f_ix + 1, 1]]).reshape(-1, 1)))

	X_train, X_test, y_train, y_test = train_test_split(modified_dataset_X, modified_dataset_y, test_size=0.2, shuffle=False)
	y_train = np.array([[1 if i + 1 == x else 0 for i in range(ACTION_NUM)] for x in y_train])
	return X_train, y_train, X_test, y_test

def load_data_window(filename, window_size = 50):
	df = pd.read_csv(filename, sep=' ', header=None)
	dataset = df.values.astype(int)
	dataset = dataset[:, :]
	modified_dataset_X, modified_dataset_y = (np.array([]), np.array([]))
	for i in range(dataset.shape[0] - window_size - 1):
		if modified_dataset_X.size == 0:
			modified_dataset_X = dataset[i:i + window_size, :].reshape(1, window_size, dataset.shape[1])
			modified_dataset_y = np.array([dataset[i + window_size + 1, 1]]).reshape(-1, 1)
		else:
			modified_dataset_X = np.vstack((modified_dataset_X, dataset[i:i + window_size, :].reshape(1, window_size, dataset.shape[1])))
			modified_dataset_y = np.vstack((modified_dataset_y, np.array([dataset[i + window_size + 1, 1]]).reshape(-1, 1)))
		# print (modified_dataset_X.shape, modified_dataset_y.shape)
	X_train, X_test, y_train, y_test = train_test_split(modified_dataset_X, modified_dataset_y, test_size=0.2, shuffle=False)
	y_train = np.array([[1 if i + 1 == x else 0 for i in range(ACTION_NUM)] for x in y_train])
	return X_train, y_train, X_test, y_test

def build_model():
	model = Sequential()
	model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(SpatialDropout1D(0.2))
	model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
	model.add(Dropout(0.1))
	model.add(LSTM(20, return_sequences=False, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(ACTION_NUM, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def evaluate_model(predicted, y_orig):
	predicted_hot = np.argmax(predicted, axis=1)
	cnt = 0
	for i in range(y_orig.shape[0]):
		print (y_orig[i][0], predicted_hot[i])
		if (y_orig[i][0] == predicted_hot[i]):
			cnt += 1
	cnt /= y_orig.shape[0]
	return cnt

def evaluate_model_two(predicted_t, y_orig):
	# print (predicted_t)
	# print (y_orig)
	predicted = np.copy(predicted_t)
	cnt = 0
	for i in range(y_orig.shape[0]):
		if y_orig[i][0] == np.argmax(predicted[i]) + 1:
			cnt += 1
			print ('true:', y_orig[i][0], 'pred:', np.argmax(predicted[i]) + 1, ' match1. confidence: ', np.max(predicted[i]))
		else:
			predicted[i][np.argmax(predicted[i])] = 0
			if y_orig[i][0] == np.argmax(predicted[i]) + 1:
				cnt += 1
				print ('true:', y_orig[i][0], 'pred:', np.argmax(predicted[i]) + 1, ' match2. confidence: ', np.max(predicted[i]))
			else:
				print ('true:', y_orig[i][0], 'pred', np.argmax(predicted_t[i]) + 1, ' fail.   confidence: ', np.max(predicted_t[i]))
	cnt /= y_orig.shape[0]
	return cnt

def plot_data_dist(y_train):	
	y_train = np.argmax(y_train, axis = 1)
	from collections import Counter
	distX = []
	distY = []
	for i in Counter(y_train.reshape(y_train.shape[0]).tolist()).items():
	    distX.append(i[0] + 1)
	    distY.append(i[1])
	plt.scatter(distX, distY)
	plt.xlabel('action')
	plt.ylabel('number of actions')
	plt.title('occurences of actions')
	plt.show()

if __name__ == "__main__":
	
	# X_train, y_train, X_test, y_test = load_data_window('balanced_final_dataset.csv', 40)
	# X_train, y_train, X_test, y_test = load_data('dataset_vgg_ast_aty_merged.csv', 2)
	X_train, y_train, X_test, y_test = load_data('dataset_30_actions_fcb_rma_full.csv', 2)
	# plot_data_dist(y_train)
	print (X_train.shape, y_train.shape)
	print (X_test.shape, y_test.shape)
	model = build_model()
	start_time = time.time()
	model.fit(X_train, y_train, epochs=200, verbose=2, validation_split = 0.0)
	model.save('lstm2_model_opencv_balanced_dataset_200e_fcb_rma.h5')  # creates a HDF5 file 'my_model.h5'
	print ("Training time: " + str(time.time() - start_time))
	start_time = time.time()
	predicted = model.predict(X_test)
	print ("Prediction time: " + str(time.time() - start_time))
	acc = evaluate_model_two(predicted, y_test)
	print ('acc: ', acc * 100, '%')
	
