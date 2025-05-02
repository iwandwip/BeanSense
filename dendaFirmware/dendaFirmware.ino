#include "library.h"

// =============== SETUP ===============
void setup() {
  Serial.begin(115200);
  setupDisplay();
  setupSensor();
  setupRTOS();
  setupFileSystem();
  displayHome();
}

// =================== LOOP ===================
void loop() {
  button.loop();
  guiHome();
}
