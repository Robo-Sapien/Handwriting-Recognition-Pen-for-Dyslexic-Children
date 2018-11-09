import time
import sys
import serial

arduino_sPort=serial.Serial('com10',115200) #com port and baud rate
dataFileHandle=open('a19.txt','w')

forceLim=5
pauseLim=100
endLim=10000
counter=0
timepoint="1t6"
maxExample=10
example=0
print "Sleeping for 10 sec"
time.sleep(10)
print "Starting to Record in 1 sec"
time.sleep(0.5)
arduino_sPort.reset_input_buffer()
while(1):
    if(arduino_sPort.inWaiting()>0):
        timepoint=arduino_sPort.readline()
        #Writing the training timeseries to file.(Single Example)
        dataFileHandle.write(timepoint.strip('\n'))

    #Breaking for the new training example, if we have stopped writing.
    split=timepoint.split('t')
    try:
        thumbForce=int(split[1][0])
    except:
        thumbForce=0 #It wont matter if one of the timepoint is random. It will get over it.
    if(thumbForce<forceLim):
        counter=counter+1
    else:
        counter=0
    #print "timePoint",timepoint
    print "counter",counter

    #Changing the trainign Example.
    if(counter>=pauseLim):
        example=example+1
        counter=0
        timepoint="1t6"
        # Ending the training Session(Multiple end will not come)
        if(example==maxExample):
            dataFileHandle.write("END\n")
            print "Training Example END"
            dataFileHandle.close()
            sys.exit()
        dataFileHandle.write("END\n")
        print "Pausing for 2 second"
        time.sleep(5)
        arduino_sPort.reset_input_buffer()
        print "Restarting in 1 sec"
        time.sleep(0.5)
