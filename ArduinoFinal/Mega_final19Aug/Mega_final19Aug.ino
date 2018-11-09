 String timePoint="";
char tempChar;

void setup(){
  Serial1.begin(115200);
  Serial.begin(115200);
  delay(1000);
  //Serial1.print(1);
  delay(100);
  Serial1.flush();
}

void loop(){
  if(Serial1.available()>0){
    tempChar=Serial1.read();
    //Serial.println(tempChar);
    timePoint=timePoint+tempChar;
  }
  if(tempChar=='K'){
    Serial.println(timePoint);
    timePoint="";
    tempChar='g';
  }
}

