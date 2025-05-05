#include <Arduino.h>

void TaskReadSensor(void *pvParameters) {
  for (;;) {
    readSensor();
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }
}


void TaskServerHandler(void *pvParameters) {
  setupServer();
  for (;;) {
    serverHandler(logicServer);
    if (Serial.available()) {
      String msg = Serial.readStringUntil('\n');
      msg.toLowerCase();
      if (msg == "r") {
        ESP.restart();
      }
    }
  }
}


// void mainTask(void *pvParameters) {
//   for (;;) {
//     button.loop();
//     guiHome();

//   }
// }



void setupRTOS() {
  Serial.begin(115200);

  xTaskCreatePinnedToCore(
    TaskReadSensor,
    "TaskReadSensor",
    10000,
    NULL,
    1,  // Prioritas lebih rendah
    NULL,
    0);

  xTaskCreatePinnedToCore(
    TaskServerHandler,
    "TaskServerHandler",
    10000,
    NULL,
    0,  // Prioritas lebih rendah
    NULL,
    0);

  // xTaskCreatePinnedToCore(
  //   mainTask,
  //   "mainTask",
  //   10000,
  //   NULL,
  //   0,     // Prioritas lebih rendah
  //   NULL,
  //   1
  // );
}
