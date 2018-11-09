import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

dfA=pd.read_excel("A.xlsx")#We have to save them with index to read the same index back. Check
dfC=pd.read_excel("C.xlsx")

def create_matrix(datArr):
	nanArr=np.isnan(datArr)
	nanArr=nanArr.reshape(-1,8).T
	matrix=datArr.reshape(-1,8).T#will automatically infer the other dimension
	
	for j in range(matrix.shape[1]):
		if(nanArr[0,j]==0):
			flag=check_completeness(nanArr,j)
			if(flag==1):
				print (j,matrix[:,j])
				sys.exit("Incomplete Value in timepoint")
		else:
			matrix=matrix[:,0:j]
			break
	#print ("mean :",np.mean(matrix[3,:]))
	matrix=endBackEffect(matrix)
	matrix=removeInconsistancy(matrix)
	#matrix=smoothen(matrix)
	return matrix

def check_completeness(isnan,j):
	for i in range(8):
		if(isnan[i,j]==1):
			return 1
	return 0

def endBackEffect(matrix):
	time=matrix.shape[1]
	meanForce=np.mean(matrix[3,:])
	thresh=meanForce//2
	threshGen=400#can be tuned
	
	for j in reversed(range(time)):
		#We could use better measure to make threshhold
		if(matrix[3,j]>threshGen):
			break
		else:
			continue
	matrix=matrix[:,0:j]
	return matrix

#Some Error is here. Correct before using(all graphs come same)
def smoothen(matrix):
	#Apply gaussian filter
	smoothLen=3#keep odd
	time=matrix.shape[1]
	pad=np.zeros((8,smoothLen//2))

	matrixN=np.zeros((8,time))
	matrix=np.concatenate((pad,matrix,pad),axis=1)
	
	halfWay=smoothLen//2
	weight=np.exp((np.arange(-1,1.1,1/halfWay)**2)/(-2))/(np.sqrt(2*np.pi))
	weight=weight.reshape(1,-1)
	
	for j in range(time):
		#print (j,j+smoothLen-1)
		matrixN[:,j]=np.sum(matrix[:,j:j+smoothLen]*weight)
	return matrixN

def removeInconsistancy(matrix):
	time=matrix.shape[1]
	default=np.mean(matrix,axis=1,keepdims=True)
	#print (default.shape,matrix.shape)
	for j in range(time):
		if(matrix[0,j]>1024 or matrix[0,j]<0):
			matrix[0,j]=min(1024,default[0])
		if(matrix[1,j]>1024 or matrix[1,j]<0):
			matrix[1,j]=min(1024,default[1])
		if(matrix[2,j]>1024 or matrix[2,j]<0):
			matrix[2,j]=min(1024,default[2])
		if(matrix[3,j]>1024 or matrix[3,j]<0):
			matrix[3,j]=min(1024,default[3])
			#print (default[3])

		# For unit quaternions the range of each element will be from -1 to 1 i guess still lets take 10 as lim
		if(matrix[4,j]>1 or matrix[4,j]<-1):
			matrix[4,j]=default[4]
		if(matrix[5,j]>1 or matrix[5,j]<-1):
			matrix[5,j]=default[5]
		if(matrix[6,j]>1 or matrix[6,j]<-1):
			matrix[6,j]=default[6]
		if(matrix[7,j]>1 or matrix[7,j]<-1):
			matrix[7,j]=default[7]
	return matrix


#Dividing the linear timePoint Series in Matrix and cleaning it(not Smoothened cuz getting bad)
dfs=[dfA,dfC]
plt.figure(1)
lengths=[] #lenght of axis 1 in each example(time point lenght)
examples=[]
tags=[]#has integer label
i=0
for df in dfs:
	j=0
	i+=1
	for cols in df:
		j+=1
		label=int(str(cols).split('.')[0])
		data=df[cols].values		#Returns a numpy array
		matrix=create_matrix(data)
		
		#print (i,j)
		if(j==1 and i==1):
			examples=[matrix]
			lengths=[matrix.shape[1]]
			tags=[label]
		else:
			examples.append(matrix)
			lengths.append(matrix.shape[1])
			tags.append(label)

		if(j%30==0 and i==2):
			plt.subplot(211)
			x=np.arange(0,matrix.shape[1])
			plt.plot(x,matrix[3,:])


'''
Now according to the lowest lenght take the first f(low_leng) frequencies
as we will have to take similar in later tests.(but keep distribution same)
And taking current lowest leght amount of frequency may cause problem later)
'''
#Starting the FFT procedure
def getDistinctNumbers(fMatrix,maxF):
	#maxF is half of allowed max freq.
	slicefM=fMatrix[:,0:maxF]
	#Removing the first imaginary coefficient as its zero and causing problem in net
	fMatrixN=np.concatenate([slicefM.real,slicefM.imag[:,1:slicefM.shape[1]]],axis=1)
	return fMatrixN

lowLength=min(lengths)
print (lowLength,max(lengths)) #lowest=63 and max=520
#lets go with top 50 frequency(I think 63 will be minimum_bounds)
maxF=50//2 #Could be changed later
fftExamples=[]
for i in range(len(tags)):
	fMatrix=np.fft.fft(examples[i],axis=1)
	leng=fMatrix.shape[1]
	#Back part sticked first(-ve frequency) then the front frequencies(+ve counterpart)
	#Though for real values they are conjugate.
	#Later replace them with imaginary part of +ve frequency
	#Automatically getting odd number of frequencies. See copy.
	#fMatrixN=np.concatenate([fMatrix[:,length-maxF+1:length],fMatrix[:,0:maxF]],axis=1)#(DONT USE)
	fMatrixN=getDistinctNumbers(fMatrix,maxF)
	if(i==0):
		fftExamples=[fMatrixN]
	else:
		fftExamples.append(fMatrixN)


print (fMatrixN.shape)
#print (length,length-maxF+1,maxF)
#for i in range(49):
	#print (i,fMatrixN[0,i])



'''
Make Series and make DataSet
'''
dataSet=pd.DataFrame()
for i in range(len(tags)):
	oneLiner=np.squeeze(fftExamples[i].reshape(1,-1))
	series=pd.Series(oneLiner)
	df=pd.DataFrame(series,columns=[tags[i]])
	dataSet=pd.concat([dataSet,df],axis=1,join='outer')

print(dataSet.shape)
dataSet.to_excel('dataSet.xlsx')
dataSet.to_csv('dataSet.csv')


#plot later with ramndom permutation
#plt.show()
