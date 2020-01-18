import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lstm2 import *

# dataset = pd.read_csv('dataset_action_1.csv', sep=' ', header=None).values.astype(int)
# action_1_ds = []
# for idx, i in enumerate(dataset):
# 	if i[1] == 1:
# 		action_1_ds = i
		# print (idx)
		# break


X_train, y_train, X_test, y_test = load_data('dataset_action_1.csv', 2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
model = build_model()
start_time = time.time()
model.fit(X_train, y_train, epochs=200, verbose=2, validation_split = 0.0)
model.save('lstm2_model_test.h5')  # creates a HDF5 file 'my_model.h5'
print ("Training time: " + str(time.time() - start_time))
start_time = time.time()
predicted = model.predict(X_test)
print ("Prediction time: " + str(time.time() - start_time))
acc = evaluate_model_two(predicted, y_test)
print ('acc: ', acc * 100, '%')