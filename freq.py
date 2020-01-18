import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data_dist(y_train):	
	print (y_train)
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



df = pd.read_csv('dataset_30_actions_fcb_rma_full.csv', sep=' ', header=None)
dataset = df.values
print (dataset.shape)
# plot_data_dist(dataset[:, 1].reshape(-1, 1).astype(int))
x = dataset[:, 1].reshape(-1, 1).astype(int)

dist = np.zeros((1, 30))


for i in x:
	dist[0, i - 1] += 1


y = dist
z = np.arange(1,31)

print (dist.astype(int))

plt.scatter(z, dist)
plt.xlabel('action')
plt.ylabel('number of actions')
plt.title('occurences of actions')
plt.show()


