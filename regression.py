import tensorflow as tf
import numpy as np
from tensorflow import keras
import csv

x1, ys = [], []

#data in .csv file minimized to range 0-1 by dividing house size by 3000, BHKs by 3 and price by 500 
with open('./house.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    for row in csv_reader:
    	if line > 0:
    		x = float(row[1]) + float(row[3])
	    	x1.append(x)
	    	ys.append(row[5])
    	line += 1


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
x1 = np.asarray(x1, dtype=float)
ys = np.asarray(ys, dtype=float)
model.fit(x1, ys, epochs=500)

while True:
	house_size = float(input('Enter the house size: '))
	house_size = house_size/3000
	bhks = float(input('Enter the BHK: '))
	bhks = bhks/3
	x = house_size + bhks
	value = model.predict([x]) * 500
	value = int(value[0][0])
	print("The estimate cost of the flat would be " + str(value) + " lakhs")
