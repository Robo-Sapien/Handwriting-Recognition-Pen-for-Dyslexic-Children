import pandas as pd
import numpy as np

h1=open('a1.txt','r')
h2=open('a2.txt','r')
h3=open('a3.txt','r')
h4=open('a4.txt','r')
h5=open('a5.txt','r')
h6=open('a6.txt','r')
h7=open('a7.txt','r')
h8=open('a8.txt','r')
h9=open('a9.txt','r')
h10=open('a10.txt','r')
fileHandles=(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10)

def parseTimePoint(timepoint,time):
    tagTup=['w','x','y','z','X','Y','Z','t']
    iteratable=[[time],tagTup]
    indexTimepoint=pd.MultiIndex.from_product(iteratable)
    #Have to make it one dimensional for converting it to Series
    value=np.empty((8),dtype=float)

    timepoint=timepoint.strip()
    try:
        timepoint=timepoint.split('w')
        timepoint=timepoint[1].split('x')
        value[0]=float(timepoint[0])

        timepoint=timepoint[1].split('y')
        value[1]=float(timepoint[0])

        timepoint=timepoint[1].split('z')
        value[2]=float(timepoint[0])

        timepoint=timepoint[1].split('X')
        value[3]=float(timepoint[0])

        timepoint=timepoint[1].split('Y')
        value[4]=float(timepoint[0])

        timepoint=timepoint[1].split('Z')
        value[5]=float(timepoint[0])

        timepoint=timepoint[1].split('t')
        value[6]=float(timepoint[0])

        timepoint=timepoint[1].split('b')
        value[7]=float(timepoint[0])
    except:
        return pd.Series([])
    series=pd.Series(value,index=indexTimepoint)
    return series

def mergeIntoDataset(dataFileHand,yLabel,dfDataset):
    timeIndex=0
    exIndex=0
    dfEx=pd.DataFrame()
    for timepoint in dataFileHand:
        timepoint=timepoint.strip()
        if(timepoint=="END"):
            dfDataset=pd.concat([dfDataset,dfEx],axis=1,join='outer')
            dfEx=pd.DataFrame()
            timeIndex=0
            exIndex+=1
            print("Parsed example "+str(exIndex))
            continue
        series=parseTimePoint(timepoint,timeIndex)
        if(len(series)==0):
            continue
        else:
            #We have to assign same column name to keep index same
            df=pd.DataFrame(series,columns=[yLabel])
            dfEx=pd.concat([dfEx,df],axis=0)
            timeIndex+=1
    return dfDataset

dfDataset=pd.DataFrame()
for file in fileHandles:
    dfDataset=mergeIntoDataset(file,3,dfDataset)
    print("File Parsing COMPLETED")


print (dfDataset.shape)
dfDataset.to_excel('C.xlsx',sheet_name='Sheet1')
dfDataset.to_csv('C.csv')
