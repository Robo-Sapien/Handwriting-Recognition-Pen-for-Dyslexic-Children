import numpy as np 

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A

def relu(Z):
	A=np.maximum(0,Z)
	return A

def tanh(Z):
	A=np.tanh(Z)
	return A

def sigmoid_backward(dA,Z):
	a=1/(1+np.exp(-Z))
	dZ=dA*a*(1-a)

	return dZ

def relu_backward(dA,Z):
	dZ=np.array(dA,copy=True)
	#Gradient is dA*1 when Z>0 and 0 elsewhere
	dZ[Z<=0]=0

	return dZ