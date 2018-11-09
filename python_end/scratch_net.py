import numpy as np
from utils import *

class model():

	def __init__(self,layers,dims,init_type,activation_tup,lambd):
		'''layers :means the number of layer in the network
			dims  :length of each layer (including the input layer)
			init_type: "HE/modifiedHE" or "XAVIER"
			activation_tup: the activation to be used for each layer
			lambd :Regularisation parameter

		'''
		self.layers=layers #excluding the input layer as by convention
		self.dims=dims
		self.parameters=self.initialize_parameters(dims,init_type)
		self.init_type=init_type
		self.cache={}
		self.cost=0
		self.activation_tup=activation_tup
		self.grads={}
		self.lambd=lambd 

	def initialize_parameters(self,dims,init_type):
		''' INPUT 	: dims=dimension of each layer list or tuple
			OUTPUT  : parameters Dictionary
		'''
		parameters={}
		num_layers=self.layers

		for l in range(num_layers):
			#He works best with relu activations in the layers(Why?)
			if(init_type=="HE"):
				#"Currently using Modified HE ie multiplying with 0.5 extra"
				parameters["W"+str(l+1)]=np.random.randn(dims[l+1],dims[l])*np.sqrt(2/dims[l])*0.5
				parameters["b"+str(l+1)]=np.zeros((dims[l+1],1))

			elif(init_type=="Xavier"):
				parameters["W"+str(l+1)]=np.random.randn(dims[l+1],dims[l])*np.sqrt(1/dims[l])
				parameters["b"+str(l+1)]=np.zeros((dims[i+1],1))
			
		return parameters

	def forward_propagate(self,X,Y):
		''' X: Training Data with examples in column 
			Y: Training label with examples in column
		'''

		cache={}
		num_layers=self.layers
		parameters=self.parameters
		activation=self.activation_tup
		cache["A"+str(0)]=X

		#Looping through the layer computing the values.
		for l in range(num_layers):
			cache["Z"+str(l+1)]=np.dot(parameters["W"+str(l+1)],cache["A"+str(l)])+parameters["b"+str(l+1)]
			if(activation[l]=="relu"):
				cache["A"+str(l+1)]=relu(cache["Z"+str(l+1)])
			
			elif(activation[l]=="sigmoid"):
				cache["A"+str(l+1)]=sigmoid(cache["Z"+str(l+1)])
		self.cache=cache

	def calculate_cost(self,Y):
		batch_size=Y.shape[1]
		layers=self.layers
		lambd=self.lambd
		AL=self.cache["A"+str(layers)]
		parameters=self.parameters

		log_probs=Y*np.log(AL)+(1-Y)*np.log(1-AL)
		cost=(-1/batch_size)*np.squeeze(np.sum(log_probs))

		L2_regularization_cost=0
		for l in range(1,layers+1):
			L2_regularization_cost=L2_regularization_cost+np.sum(np.square(parameters["W"+str(l)]))

		L2_regularization_cost=(lambd/(2*batch_size))*L2_regularization_cost

		cost=cost+L2_regularization_cost

		return cost

	def back_propagate_layer(self,m,layer):
		'''INPUT:  m: is batch size
			   layer: is exact layer number
		'''
		lambd=self.lambd
		activation=self.activation_tup[layer-1] #len(activation_tup) is total layer number-1

		dA=self.grads["dA"+str(layer)]
		Z=self.cache["Z"+str(layer)]
		A_prev=self.cache["A"+str(layer-1)]
		W=self.parameters["W"+str(layer)]

		#Activation Backpropagate
		if(activation=="sigmoid"):
			dZ=sigmoid_backward(dA,Z)
		elif(activation=="relu"):
			dZ=relu_backward(dA,Z)
		self.grads["dZ"+str(layer)]=dZ

		#Linear Backpropagate
		self.grads["dW"+str(layer)]=(1/m)*np.dot(dZ,A_prev.T)+(lambd/m)*W #with regularisation term
		self.grads["db"+str(layer)]=(1/m)*np.sum(dZ,axis=1,keepdims=1)

		self.grads["dA"+str(layer-1)]=np.dot(W.T,dZ)

	def back_propagate_model(self,Y):
		''' INPUT: Y: Training Label
			OUTPUT: updates the model varible grads
		'''
		layers=self.layers
		m=Y.shape[1]

		AL=self.cache["A"+str(layers)]
		self.grads["dA"+str(layers)]=np.divide(-1*Y,AL)+np.divide(1-Y,1-AL)

		for l in reversed(range(1,layers+1)):
			self.back_propagate_layer(m,l)
			#print (self.grads["dW"+str(l)].shape,self.grads["db"+str(l)].shape)

	def param_to_vector(self):

		params=[]
		layers=self.layers
		for l in range(layers):
			params.append(self.parameters["W"+str(l+1)])
			params.append(self.parameters["b"+str(l+1)])
		#print (params)
		count=0
		for param in params:
			#print (param)
			new_vector=np.reshape(param,(-1,1))
			if(count==0):
				theta=new_vector
			else:
				theta=np.concatenate((theta,new_vector),axis=0)
			count=count+1
		return theta

	def grads_to_vector(self):
		
		#We have to bring the theta and grad in same order to make sense
		grads=[]
		layers=self.layers
		for l in range(layers):
			grads.append(self.grads["dW"+str(l+1)])
			grads.append(self.grads["db"+str(l+1)])
		#print(grads)
		count=0
		for grad in grads:
			#print (grad)
			new_vector=np.reshape(grad,(-1,1))
			if(count==0):
				gradient=new_vector
			else:
				gradient=np.concatenate((gradient,new_vector),axis=0)
			count=count+1
			
		return gradient

	def vector_to_param(self,theta):

		layers=self.layers
		parameters={}

		start=0
		for l in range(1,layers+1):
			n1,n2=self.parameters["W"+str(l)].shape
			end=start+n1*n2
			parameters["W"+str(l)]=theta[start:end].reshape((n1,n2))
			start=end
			end=start+n1
			parameters["b"+str(l)]=theta[start:end].reshape((n1,1))
			start=end
		return parameters

	def gradient_checking(self,epsilon,X,Y):

		param_vector=self.param_to_vector()
		gradient_vector=self.grads_to_vector()
		num_param=param_vector.shape[0]
		J_plus=np.zeros((num_param,1))
		J_minus=np.zeros((num_param,1))
		grad_approx=np.zeros((num_param,1))

		for i in range(num_param):

			#Calculating the right sided value for Central Difference
			theta_plus=np.copy(param_vector) #fresh memory (new clone) so that we could change
			theta_plus[i][0]=theta_plus[i][0]+epsilon
			self.parameters=self.vector_to_param(theta_plus)
			self.forward_propagate(X,Y)
			J_plus[i][0]=self.calculate_cost(Y)

			#Calculating the left side value for Central Differnce
			theta_minus=np.copy(param_vector)
			theta_minus[i][0]=theta_minus[i][0]-epsilon
			self.parameters=self.vector_to_param(theta_minus)
			self.forward_propagate(X,Y)
			J_minus[i][0]=self.calculate_cost(Y)

			grad_approx[i][0]=(J_plus[i][0]-J_minus[i][0])/(2*epsilon)

		numerator=np.linalg.norm(gradient_vector-grad_approx)
		denominator=np.linalg.norm(gradient_vector)+np.linalg.norm(grad_approx)

		difference=numerator/denominator
		print ("difference in grad",difference)

		#Re-assigning the initial parameters as model parameters
		self.parameters=self.vector_to_param(param_vector)

	def update_parameter(self,learningRate):
		'''
		Updating the parameter according to the gradient 
		calculated above.
		'''
		layers=self.layers
		parameters=self.parameters
		grads=self.grads

		for l in range(1,layers+1):
			parameters["W"+str(l)]=parameters["W"+str(l)]-learningRate*grads["dW"+str(l)]
			parameters["b"+str(l)]=parameters["b"+str(l)]-learningRate*grads["db"+str(l)]


		