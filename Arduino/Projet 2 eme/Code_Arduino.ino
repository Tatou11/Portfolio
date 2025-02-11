#include <TimerOne.h>
#include <Wire.h>
#include <MultiFuncShield.h>
///////////////////////////////////////////////Variable for the sonar
const int pingPin = 31; // Trigger Pin of Ultrasonic Sensor
const int echoPin = 30; // Echo Pin of Ultrasonic Sensor
float duration;
float cvm;

////////////////////////////////////////////Variable for the detection sensor
bool detect = true;
bool en_detection = false;

const int ProxSensor = 52; //Declaring where the Out pin from the sensor is wired
////////////////////////////////////////////Variable for the calibration of Mister sonar (//in cm)
const float Cal31 = 3.5;
const float Cal32 = 2.5;
const float Cal21 = 4.75;
const float Cal22 = 3.51;
const float offset = 6.5;
//
const float BaseValue1 = 4.98;
const float BaseValue2 = 4.28;
const float BaseValue3 = 3.09;
////////////////////////////////////////////Variable for the programme


int nbBlock = 0; //number of block
int nbPlaque = 0; //number of plates (Present)
//current
int nbPlaqueActuel = 0; //number of plates 

char varAffiche[4]; //variable for display
String SuperVar;
String Separ = "|";

////////////////////////////////////////////Variable for calculating the average
int i = 0;
float MoyenneTemp = 0;
float MoyenneTrue = 0;
bool CalculeEffect = false;

void setup() {
  Serial.begin(9600); // Starting Serial Terminal
  //pouir le mfs
  Timer1.initialize();
  MFS.initialize( & Timer1); // initialize multi-function shield library
  pinMode(ProxSensor, INPUT); // then we have the out pin from the module 
}

void loop() {
 
  //SonnarAffiche();
  if (digitalRead(ProxSensor) == HIGH) //Check the sensor output if it's high
  {
    delay(50);//serves to slow down the arduino, because if it goes too fast there are false positives
    detect = false;
    en_detection = false;
    MoyenneTrue = 0;
    i = 0;
    CalculeEffect = false;
    MoyenneTemp = 0;
  } else {
    delay(50);//serves to slow down the arduino, because if it goes too fast there are false positives
    detect = true;

  }

  //1 One block or more blocks are detected with the detection sensor
  if (detect == true) {
    if (en_detection == false) {
      nbBlock = nbBlock + 1; 
      en_detection = true;
    }
  }
  if ((detect == true) && (en_detection == true) && (CalculeEffect == false)) 
  {
    while (i < 50) {//2 Detects distance with an average of 50 measurements
      cvm = Sonnar();
      MoyenneTemp = MoyenneTemp + cvm;
      i++;
    }
    CalculeEffect = true;
    MoyenneTrue = MoyenneTemp / 50;
    if (Cal31 >= MoyenneTrue && Cal32 <= MoyenneTrue) //3
    {
      nbPlaque = nbPlaque + 3;
      nbPlaqueActuel = 3;
    } else {
      if (Cal21 >= MoyenneTrue && Cal22 <= MoyenneTrue) //2
      {
        nbPlaque = nbPlaque + 2;
        nbPlaqueActuel = 2;
      } else {
        nbPlaque = nbPlaque + 1;
        nbPlaqueActuel = 1;
      }

    }
     //3 Data is transferred to the RP

   SuperVar = (String(nbPlaqueActuel) + Separ + String(cvm) + Separ + String(offset));    
   Serial.println(SuperVar);
   delay(50);//for security on the serial port
 
  }
  varAffiche[0] = nbBlock + 48;
  varAffiche[1] = 45;
  varAffiche[2] = ((nbPlaque / 10) + 48);
  varAffiche[3] = ((nbPlaque % 10) + 48);
  MFS.write(varAffiche, 3);



  //4 We have a button to reset the variables 
  if (digitalRead(A1) == 0) 
  {
    Button();
  }
}
void Button() {
  nbPlaqueActuel = 0;
  nbBlock = 0;
  nbPlaque = 0;
  
}

long microsecondsToCentimeters(long microseconds) {
  long test;
  test = microseconds / 29 / 2;
  return test;
}
//returns the distance of the object from the sensor
float Sonnar() {
  pinMode(pingPin, OUTPUT);
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(pingPin, LOW);
  pinMode(echoPin, INPUT);
  duration = (pulseInLong(echoPin, HIGH));
  return ((duration) * 340 / 20000);
}
