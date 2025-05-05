#include <TFT_eSPI.h>          // lcd tft
#include <ezButton.h>          // button
#include <SPIFFS.h>            // file system
#include <FS.h>                // file system
#include <Wire.h>              // i2c
#include <DHT.h>               // dht
#include <Adafruit_ADS1X15.h>  // sensor adc to read mq
#include <WiFi.h>              // wifi
#include <WebServer.h>         // server


//==================server variabel=======================
const char* ssid = "Denda Kopi Classification";
const char* password = "123456789";
bool logicServer = false;
volatile bool commandReady = false;
String commandToSend = "";
String lastAccuracy = "";

const char* commandLabel[2] = {};





//================ sensor variabel ==============================

bool readLogic = false;
#define DHT_SENSOR_PIN 4
#define DHT_SENSOR_TYPE DHT22
#define NUM_SENSORS 8
#define NUM_READINGS 20
#define MAX_SENSORS 8
float humidity, temperature;
int sensorValues[NUM_SENSORS];



// =============== ENUM STATE ===============
enum AppState { MENU,
                SELECT_DATASET,
                EDIT_DATASET };
AppState currentState = MENU;


//================SPIDER GRAPH===============================
#define MAX_SENSORS 8
int NUM_SENSOR[3] = { 4, 6, 8 };
int countDataLoad;
String tempData[8];
const char* sensorLabels[8] = { "MQ135", "MQ2", "MQ3", "MQ6", "MQ138", "MQ7", "MQ136", "MQ5" };
#define RADIUS 50
#define CENTER_X 100
#define CENTER_Y 85
#define MIN_VAL 0
#define MAX_VAL 8000


const int screenWidth = 220;
const int screenHeight = 135;
const int barHeight = 20;
const int barWidthMax = 200;
const int barX = (screenWidth - barWidthMax) / 2 + 50;
const int barY = (screenHeight - barHeight) / 2 + 50;





// =============== PIN ===============
#define CLK_PIN 25
#define DT_PIN 26
#define SW_PIN 27
#define pumpInhale 23
#define pumpFlush 32

// =============== CONSTRUCTOR  ===============

bool prev_CLK_state;


//==========================LOGIC PUMP============================
bool pompaNoseLogic = false;
bool pompaFlushLogic = false;

// =============== MENU UTAMA ===============
const char* menuItems[] = { "Train Dataset", "Predict Data", "Edit Dataset", "Add Dataset" };
const int totalMenuItems = 4;
int menuIndex = 0;


// =============== SUBMENU PILIH DATASET ===============
const char* datasetFiles[] = { "/dataset4.csv", "/dataset6.csv", "/dataset8.csv" };
const char* sensorName[] = { "4 Sensor", "6 Sensor", "8 Sensor" };
const int totalDatasets = 3;
int datasetIndex = 0;

String selectedDataset = "";

// =============== SCROLL DATASET ===============
int scrollOffset = 0;
int totalLines = 0;
const int linesPerPage = 10;
int selectedLine = 0;

//================= variabel method =============================
const char* methods[] = {
  "AdaBoost", "CatBoost", "LGBM-ResNet",
  "LGBM-MobileNet", "LGBM-ICCS",
  "RBF-SVM_GS",
  "LGBM-Default"
  
};
int methodIndex = 0;
const int totalMethods = 7;
const int methodsPerPage = 4;
const int methodPages = (totalMethods + methodsPerPage - 1) / methodsPerPage;


//================construc all linrary=========================
DHT dht_sensor(DHT_SENSOR_PIN, DHT_SENSOR_TYPE);
Adafruit_ADS1115 ads1;  // 0x48
Adafruit_ADS1115 ads2;  // 0x49
TFT_eSPI tft = TFT_eSPI();
ezButton button(SW_PIN);
WebServer server(80);
