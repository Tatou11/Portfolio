
#include <SPI.h>
#include <RFID.h>
#include <TM1637Display.h>

#include <Pinger.h>
#include <PingerResponse.h>
#include <ESP8266WiFi.h>


const int CLK = 4; 
const int DIO = 16;
TM1637Display display(CLK, DIO);

#define SS_PIN 2
#define RST_PIN 0

RFID rfid(SS_PIN, RST_PIN);


const char* ssid = "PC1";
const char* password = "11111111";
const char* host = "192.168.137.1";
String donnee_recue = "";
bool syncro = true;
Pinger pinger;
WiFiClient client;


String tag_Temps = "";

int const nbCarteMax = 27;

String Tableaux_RFID[nbCarteMax];
String Tableaux_PerRFID[nbCarteMax];
String Tableaux_NUMRFID[nbCarteMax];

bool temp_Envoie = false;

bool acces = false;
double timer1 = 0;
double timer2 = 0;
bool lecture = false;
const int httpPort = 80;

String mdp = "EZqVtcX4571";
void setup() {

  display.setBrightness(0x0a); 

  pinMode(15, OUTPUT);
  pinMode(5, OUTPUT);

  SPI.begin();
  rfid.init();

  display.showNumberDec(0000);

  WiFiClient client;
  Serial.begin(9600);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  timer1 = millis();
  timer2 = millis();
}
void loop()
{
  if ((millis() - timer1) > 5000)
  {
    display.showNumberDec(0000);
    acces = false;
    timer1 = millis();
  }
  if ((millis() - timer2) > 3000 && temp_Envoie)
  {
    temp_Envoie = false;
  }
 

  if (acces)
  {
    digitalWrite(15, HIGH);
    digitalWrite(5, LOW);
  }
  else
  {
    digitalWrite(5, HIGH);
    digitalWrite(15, LOW);
  }
  if (!rfid.isCard())
  {
    return;
  }
  if (rfid.readCardSerial() && temp_Envoie == false)
  {
    Serial.println(timer1);
    Serial.println(timer2);
    Recevoir_Donnees();
    tag_Temps = "";
    tag_Temps = String(rfid.serNum[0], HEX) + String(rfid.serNum[1], HEX) + String(rfid.serNum[2], HEX) + String(rfid.serNum[3], HEX) + String(rfid.serNum[4], HEX);
    tag_Temps.toUpperCase();
    Serial.println("             ");
    timer1 = millis();
    for (int i = 0; i < nbCarteMax; i++)
    {
      if (tag_Temps == Tableaux_RFID[i]) 
      {
        if (Tableaux_PerRFID[i] == "true") 
        {
          Serial.println("Acces OK");
          display.showNumberDec((Tableaux_NUMRFID[i]).toInt()); 
          acces = true;
          Envoie_Donnee(Tableaux_NUMRFID[i], "true");
        }
        else//acces KO
        {
          Serial.println("Acces refusé");
          acces = false;
          Envoie_Donnee(Tableaux_NUMRFID[i], "false");
        }
      }
    }
    temp_Envoie =true ;//avant true
    timer1 = millis();
    timer2 = millis ();
    //clear
    rfid.halt();
    tag_Temps = "";
  }
}
void Envoie_Donnee(String Num, String Perm)
{
  if (WiFi.isConnected())
  {
    if (!client.connect(host, httpPort)) {
      return;
    }
    client.print(String("POST ") + "/envoieDonne.php?&Mdp=" + (mdp) + "&Num=" + (Num) + "&Acces=" + (Perm) + " HTTP/1.1\r\n" + "Host: " + host + "\r\n" + "Connection: keep-alive\r\n\r\n"); 
    client.stop();
  }
}
void Recevoir_Donnees()
{
  if (WiFi.isConnected())
  {
    if (!client.connect(host, httpPort)) {
      return;}
    client.print(String("GET ") + "/prendre.php?&Mdp=" + (mdp) + " HTTP/1.1\r\n" + "Host: " + host + "\r\n" + "Connection: keep-alive\r\n\r\n");
    client.stop();
    while (client.available()) {
      String line = client.readStringUntil('</body>');
      donnee_recue = donnee_recue + line;
      client.stop();
    }
    int j = 0;
    int taille = donnee_recue.length() + 1;
    char Buf[taille];
    donnee_recue.toCharArray(Buf, taille);
    bool important = false;
    String temps = "";
    while (j < taille)
    {
      if (Buf[j] == '#')
      {
        important = !important;
        Buf[j] = ' ';
      }
      if ( important == false)
      {
        Buf[j] = ' ';
      }
      j++;
    }
    //on retire les espaces
    String StringTemporaire = String(Buf);
    StringTemporaire.replace(" ", "");
    //retire le premier
    StringTemporaire.remove(0, 1);
    StringTemporaire.toCharArray(Buf, StringTemporaire.length());
    //traitement des données: mise dans un tableaux
    j = 0;
    int k = 0;
    int w = 0;
    int z = 0;
    taille = sizeof(Buf);
    bool permDonne = false;
    bool tag = false;
    while (j < taille)
    {
      if ((Buf[j] == ('!')))
      {
        permDonne = true;
        j = j + 2;
      }
      if ((Buf[j] == ('/')))
      {
        if (permDonne == false)
        {
          tag = !tag;
          if (tag == true)
          {
            Tableaux_RFID[k] = temps;
            k = k + 1;
          } else
          {
            Tableaux_NUMRFID[w] = temps;
            w = w + 1;
          }
          temps = "";
        }
        else
        {
          if (z < nbCarteMax) {
            Tableaux_PerRFID[z] = temps;
            z = z + 1;
            temps = "";
          }
        }
      }
      else
      {
        temps = temps + String(Buf[j]);
      }
      j++;
    }
    j = 0;
    donnee_recue  = "";
    temps = "";
    client.stop();
  }
}
