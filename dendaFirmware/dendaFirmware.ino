#include "display.h"
#include "sensor.h"
#include <SPIFFS.h>
#include <FS.h>


// =============== ENUM STATE ===============
enum AppState { MENU, SELECT_DATASET, EDIT_DATASET };
AppState currentState = MENU;


// =============== MENU UTAMA ===============
const char* menuItems[] = {"Train Dataset", "Predict Data", "Edit Dataset", "Add Dataset" };
const int totalMenuItems = 4;
int menuIndex = 0;


// =============== SUBMENU PILIH DATASET ===============
const char* datasetFiles[] = {"/dataset4.csv", "/dataset6.csv", "/dataset8.csv" };
const int totalDatasets = 3;
int datasetIndex = 0;
String selectedDataset = "";

// =============== SCROLL DATASET ===============
int scrollOffset = 0;
int totalLines = 0;
const int linesPerPage = 10;
int selectedLine = 0;

// =============== SETUP ===============
void setup() {
  Serial.begin(115200);
  setupDisplay();
  if (!SPIFFS.begin(true)) {
    tft.setCursor(20, 50);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.println("SPIFFS GAGAL!");
    while (1);
  }

  displayHome();
}
