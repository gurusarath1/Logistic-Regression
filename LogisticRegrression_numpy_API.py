# Created by Guru Sarath
# Date: 17th Aug 2019

import numpy as np

class LogisticRegression:

	X_train = None #(m , n)
	y_train = None #(m , 1)

	X_test = None #(p , n)
	y_test = None #(p , 1)

	y_pred = None
	y_pred_testSet = None

	m = 0 #Number of training data
	n = 0 #Number of features

	w = None #Weights
	b = 0 #bais

	learningRate = 0.001
	iterations = 1000000


	def __init__(self, X_train_, y_train_, X_test_ = None, y_test_ = None):

		self.X_train = X_train_
		self.y_train = y_train_
		self.X_test = X_test_
		self.y_test = y_test_

		self.m = y_train.shape[0]
		self.n = X_train.shape[1]

		self.init_weights() #Initialize weights randomly


	# Outputs probablity for a given feature vector 
	def predict(self, X):
		print(LogisticRegression.forwardProp(X,self.w,self.b))


	# Initialize weights randomly
	def init_weights(self):
		self.w = np.random.random((1, self.n))
		self.b = np.random.random()


	# Train the machine learning model
	def fit(self, learningRate = 0.001):

		j = 0
		prev_cost = np.inf
		dw = np.zeros((self.n,1))
		db = 0
		self.learningRate = learningRate

		print('Training ... ... ...')

		for i in range(self.iterations):

		    self.y_pred = LogisticRegression.forwardProp(self.X_train, self.w, self.b)
		    
		    curr_cost = LogisticRegression.costFunction(LogisticRegression.lossFunction(self.y_train, self.y_pred))
		    
		    if j == 50000:
		        j = 0
		        print(i,'  ' ,curr_cost)
		        
		    # Halt the training process if cost is increasing
		    if curr_cost > prev_cost:
		        print(i, '   Increasing  ', curr_cost)
		        break

		    # Find gradients

		    dz = self.y_pred - self.y_train
		    db = np.sum(dz) / self.m
		    for weight_i in range(self.n):
		    	dw[weight_i,0] = np.sum(X_train[:,weight_i].reshape(self.m,1) * dz) / self.m 


		    # Update Weights

		    self.b = self.b - (self.learningRate * db)

		    for weight_i in range(self.n):
		    	self.w[0,weight_i] = self.w[0,weight_i] - (self.learningRate * dw[weight_i,0])


		    prev_cost = curr_cost
		    j += 1

	# Test the machine learning model on the test dataset
	def test(self):

		self.LogisticOutput(self.X_test,self.w,self.b)
		correct = 0
		wrong = 0
		for i in range(self.y_test.shape[0]):
		    if self.y_pred_testSet[i] == self.y_test[i]:
		        correct += 1
		    else:
		        print(i, 'th prediction was wrong  X = ', X_train[i], ' Predicted = ', self.y_pred_testSet[i])
		        wrong += 1
		print('\n\nCorrect predictions = ', correct)
		print('Wrong predictions = ', wrong)
		print('Testset Accuracy = ', (correct / (correct + wrong)) * 100, '%')


	# Output of logistic unit
	def LogisticOutput(self, X,w,b):
	    self.y_pred_testSet = LogisticRegression.forwardProp(X,w,b)
	    self.y_pred_testSet = self.y_pred_testSet > 0.5
	    self.y_pred_testSet = self.y_pred_testSet.astype(int)  # Convert boolean array to integer array  #astype does not change the original array 
	
	# Sigmoid activation function
	@staticmethod
	def sigmoid(z):
		return (1 / (1 + np.exp(-z)))

	# Loss function for a single training example
	@staticmethod
	def lossFunction(y,y_pred):
		return -(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

	# Average loss for all examples
	@staticmethod
	def costFunction(loss):
		return np.sum(loss) / loss.shape[0]

	# Raw output of single logistic unit
	@staticmethod
	def forwardProp(X,w,b):
		z = np.matmul(X, w.T) + b
		y_pred = LogisticRegression.sigmoid(z)
		return y_pred


#---------------------------------------------------------------------------------------------------------

# Import Training set
DataSet = np.loadtxt('Train.csv', delimiter = ',', unpack = False, dtype = float)
X = DataSet[:,0:-1]
y = DataSet[:,-1]
X_train = X
y_train = y.reshape(y.shape[0],1)


# Import Testing set
DataSet = np.loadtxt('Test.csv', delimiter = ',', unpack = False, dtype = float)
X = DataSet[:,0:-1]
y = DataSet[:,-1]
X_test = X
y_test = y.reshape(y.shape[0],1)

# Perform logistic regression
L = LogisticRegression(X_train, y_train, X_test, y_test)
L.fit()
L.test()
L.predict([1, 2])