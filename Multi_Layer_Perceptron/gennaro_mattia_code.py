import numpy as np
import matplotlib.pyplot as plt
import random
from random import shuffle
import math

"""
def generateMatrices(x1, x2, y1, y2, c, n):

Function that takes 5 arguments: coordinate x1, coordinate x2, coordinate y1, coordinate y2, the class and the number of points.
It returns a matrix of shape (500,3)

"""
def generateMatrices(x1, x2, y1, y2, c, n):
		#create n random values in range x1,x2
		x = np.random.uniform(x1, x2, n)
		#create n random values in range y1,y2
		y = np.random.uniform(y1, y2, n)
		#create n identic values
		classes = np.repeat(c, n , axis = 0)
		#unify those values into a big array of shape (500,3) 
		array = np.c_[x, y, classes]
		#convert the array into matrix and transpose it
		m = np.asmatrix(array)
		m_tran = np.transpose(m)

		#return the transposed array
		return m_tran
"""
def rotatedMatrix(matrix, angle):

Function that takes as parameters the matrix and the angle, and computes the rotation of the given matrix.
It returns the rotated matrix

"""
def rotatedMatrix(matrix, angle):

		#compute theta
		theta = ((-angle/180) * np.pi)
		#initialise the rotation matrix
		rotation = np.array([[np.cos(theta), -np.sin(theta)],
							[np.sin(theta), np.cos(theta)]])
		#rotate the given matrix
		rotatedMatrix = np.array(np.dot(np.transpose(matrix[:2]), rotation))

		#transpose the matrix
		m_tran = np.transpose(rotatedMatrix)

		#return the matrix
		return m_tran

"""
def generateMatricesDifferent(m1, m2, x1, x2, y1, y2, c, n):

Function that creates a matrix using a different method from the one before.
It takes 8 parameters, which are: the mean, the covariance, the class and the number
It returns a matrix of shape (500,3)
"""
def generateMatricesDifferent(m1, m2, x1, x2, y1, y2, c, n):

		#initialise the mean
		mean = [m1, m2]
		#initialise the cov
		cov = [[x1, x2],[y1, y2]]
		#using the np.random.multivariate_normal() to get x and y
		x, y = np.random.multivariate_normal( mean, cov, n).T
		#create n identic values
		classes = np.repeat(c,n, axis = 0)
		#unify those values into a big array of shape (500,3) 
		array = np.c_[x, y, classes]
		#convert the array into matrix and transpose it
		m = np.asmatrix(array)
		m_tran = np.transpose(m)

		#return the transposed array
		return m_tran

"""
def concatenateMatrix(matrix1, matrix2, matrix3, matrix4):

Function that concatenates four matrices into one.
It returns a matrix of shape (2000, 3)

"""

def concatenateMatrix(matrix1, matrix2, matrix3, matrix4):

		#concatenate the matrices
		matrix = np.vstack((np.transpose(matrix1),np.transpose(matrix2),np.transpose(matrix3),np.transpose(matrix4)))

		#return the result
		return matrix

"""
def splitArray(matrix):

Function that takes a matrix as parameter and splits it into three different one.
It returns the three new matrices

"""

def splitArray(matrix):

		#coping the parameter matrix into a smaller one (25%)
		training = matrix[:500].copy()
		#coping the parameter matrix into a smaller one (25%)
		validation = matrix[500:1000].copy()
		#coping the parameter matrix into a smaller one (50%)
		test = matrix[1000:].copy()

		#return the three matrices
		return training, validation, test

"""
def saveText(matrix):

Function that saves the generated data into a .txt file
"""

def saveText(matrix):

		np.savetxt("gennaro_mattia_data.txt", matrix, delimiter = ",")

"""
def sigmoid(matrix): return 1 / (1 + np.exp(-matrix))

Function that computes the sigmoid of the matrix passed as argument

"""
def sigmoid(matrix): return 1 / (1 + np.exp(-matrix))

"""
def sigmoid_(matrix): return np.multiply(matrix, (1 - matrix))

Function that computes the derivative of the sigmoid

"""
def sigmoid_(matrix): return np.multiply(matrix, (1 - matrix))

"""
def classify_mlp(w, h, x):

Function that takes three arguments: weights, hidden layers and point.
It returns the predicted class from neural network

"""

def classify_mlp(w, h, x):

	#adding the bias to the point
	x = np.append(x, 1)
	#computing the Hidden Layer
	hLayer = sigmoid(np.dot(x,w[0]))
	#adding the Bias to the Hidden Layer
	hLayer = np.append(hLayer, 1)
	#computing the Output Layer
	oLayer = sigmoid(np.dot(hLayer, w[1]))

	#returns the index of the biggest number, which corresponds to the class
	return np.argmax(oLayer) + 1

"""
def train_mlp(w, h, eta, D):

Function that takes four parameters: weights, hidden layers, learning rate, and a matrix
It returns the updated weights

"""

def train_mlp(w, h, eta, D):

	#looping trough all the points in the matrix
	for point in D:

		p = point.copy()
		#getting the value of the class for each point
		c = p.item(2)
		#computing the Hidden Layer
		hLayer = sigmoid(np.dot(p,w[0]))
		#adding the Bias to the hidden Layer
		hLayer = np.c_[hLayer, [1]]
		#computing the Output Layer
		oLayer = sigmoid(np.dot(hLayer, w[1]))

		"""Start of backwards Propagation"""

		#inizialise the expected error as an array of 4 zeroes
		exp = np.zeros(4)
		#setting the class to be 1 in the array
		exp[int(c) -1] = 1
		#computing error
		error = exp - oLayer
		#computinh Delta Output
		dOutput = np.multiply(error, sigmoid_(oLayer))
		#computing the second error
		error2 = dOutput.dot(w[1].T)
		#computing Delta Hidden
		dHidden = np.multiply(error2, sigmoid_(hLayer))

		#updating the weights
		w[1] += hLayer.T.dot(dOutput) * eta
		w[0] += p.T.dot(dHidden[:,:h]) * eta
		
	#return the weights
	return w


"""
def evaluate_mlp(w, h, D):

Function that takes three parameters: weights, hidden layer and a matrix.
It returns the number of missclassified points

"""
def evaluate_mlp(w, h, D):

	misclassified = 0

	#looping throug the points
	for p in D:
		#getting only the x and y coordinates(first column and second one)
		point = [p.item(0), p.item(1)]
		#checking if the class of the point corresponds to the one predicted by the neural network
		if p.item(2) != classify_mlp(w, h, point):
			#if not, update the value of the missclassifie
			misclassified += 1
			
	#return the value
	return misclassified

"""
def script():

Function that calls all the previous function and plots the matrices and errors
"""

def script():

	print("----------------------------------------------------------------------------")
	print("The script will plot three different graphs, which will take a bit of time.")
	print("----------------------------------------------------------------------------")

	#intialise the matrices
	matrix = generateMatrices(2,5,1,4,1,500)
	matrix_2 = generateMatrices(1,3,-5,-1,2,500)
	matrix_3 = generateMatricesDifferent(-2,-3,0.5,0,0,3,3,500)
	matrix_4 = generateMatricesDifferent(-4,-1,3,0.5,0.5,0.5,4,500)

	#rotate the first two
	rotated1 = rotatedMatrix(matrix, 75)
	rotated2 = rotatedMatrix(matrix_2, 75)


	plt.axis([-10, 10, -10, 10])

	plt.plot(rotated1[0], rotated1[1], 'or')
	plt.plot(rotated2[0], rotated2[1], 'og')
	plt.plot(matrix_3[0], matrix_3[1], 'ob')
	plt.plot(matrix_4[0], matrix_4[1], 'ok')
	plt.figure("Function error")

	

	#concatenate the matrices into one
	finalMatrix = concatenateMatrix(matrix,matrix_2,matrix_3,matrix_4)

	#randomly shuffle the content of the matrix
	np.random.shuffle(finalMatrix)

	#save the data into the .txt file
	saveText(finalMatrix)

	#split the matrix into three smaller ones
	training, validation, test = splitArray(finalMatrix)



	#initialising the values
	x = training[0,:2]
	h = 2
	eta = 0.2

	#randomly creating the initial weights [3 is for the input layer(x,y,bias)], [4 is for the output(classes 1,2,3 and 4)]
	w0 = np.random.uniform(-0.5, 0.5 , (3, h))
	w1 = np.random.uniform(-0.5, 0.5 , (h+1, 4))
	w = [w0, w1]

	
	#Uncomment this part to see the error function plotted

	#initialise a list
	errors = []
	#loop trough the matrix
	for i in range(len(validation)):
		print("Iteration: " ,i)
		#train the mlp on the training set and update the weights
		w = train_mlp(w, h, eta, training)
		#evaluate the temporary errors using the validation set
		temp = evaluate_mlp(w, h, validation)
		#append the temporary errors in the list
		errors.append(temp)
		#getting the previous value of the list
		previous = errors[i-1]
		previous += 1
		#checking if the current value is bigger then the previous + 1
		if errors[i] > previous:
			print("Training stopped, minimum value found")
			break
		#if it doesn't increase, break the loop after iterating half of the set
		elif i >= len(training)/2:
			break

		print("After evaluation: " +str(temp))

	#getting the minimum value of the list
	stopTraining = min(errors)
	#printing when the training should stop
	print("Stop training: " ,stopTraining)
	#plot the list
	plt.plot(errors)
	#getting the prediction using the test set
	errorAfterTraining = evaluate_mlp(w ,h, test)
	print("Error predicted by the test set: ", errorAfterTraining)
	plt.plot(errorAfterTraining,'g^')

	

	for e in range(0, 3000, 3):
		w = train_mlp(w, h, eta, test)
		#getting the point coordinates
		point = [test.item(e), test.item(e+1)]
		#plotting them in different shapes depending on the prediction
		if classify_mlp(w, h, point) == 1:
			plt.plot(point[0],point[1], 'or')
		elif classify_mlp(w, h, point) == 2:
			plt.plot(point[0],point[1], 'oy')
		elif classify_mlp(w, h, point) == 3:
			plt.plot(point[0],point[1], 'ok')
		elif classify_mlp(w, h, point) == 4:
			plt.plot(point[0],point[1], 'ob')
		plt.figure("Classes predicted")
	
	#show everything that has been plotted
	plt.show()

script()
