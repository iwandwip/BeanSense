#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>

const char* ssid = "ESP32-AP";
const char* password = "123456789";

WebServer server(80);

const int buttonPin = 27;
volatile bool commandReady = false;
String commandToSend = "";
String lastAccuracy = "";

const char* commandLabel[2] = {} ; 
// Fungsi membaca file CSV dari SPIFFS
String readCSV(const char* filename) {
  File file = SPIFFS.open(filename, "r");
  if (!file) return "Failed to open file";
  String fileContent = file.readString();
  file.close();
  return fileContent;
}

// ISR tombol
void IRAM_ATTR handleButtonPress() {
  if (!commandReady) {
    commandToSend = "train,1,2";
    commandReady = true;
  }
}

void setup() {
  Serial.begin(115200);

  // Setup tombol
  pinMode(buttonPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(buttonPin), handleButtonPress, FALLING);

  // Setup SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("Failed to mount SPIFFS");
    return;
  }

  // Setup WiFi AP
  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());

  // Endpoint baca CSV
  server.on("/read-dataset4", HTTP_GET, []() {
    server.send(200, "text/plain", readCSV("/dataset4.csv"));
  });

  server.on("/read-dataset6", HTTP_GET, []() {
    server.send(200, "text/plain", readCSV("/dataset6.csv"));
  });

  server.on("/read-dataset8", HTTP_GET, []() {
    server.send(200, "text/plain", readCSV("/dataset8.csv"));
  });

  // Endpoint untuk mengirim command
  server.on("/get-command", HTTP_GET, []() {
    if (commandReady) {
      server.send(200, "text/plain", commandToSend);
      Serial.println("📤 Sent command: " + commandToSend);
      commandReady = false;
    } else {
      server.send(204, "text/plain", "");
    }
  });

  // Endpoint untuk menerima hasil akurasi
  server.on("/post-result", HTTP_POST, []() {
  Serial.println(">> POST request received");
  if (server.hasArg("plain")) {
    lastAccuracy = server.arg("plain");
    Serial.println("📥 Accuracy: " + lastAccuracy);
    server.send(200, "text/plain", "OK");
  } else {
    Serial.println("❌ No plain body received");
    server.send(400, "text/plain", "Missing plain body");
  }
});


  // Endpoint untuk melihat hasil akurasi terakhir
  server.on("/view-accuracy", HTTP_GET, []() {
    server.send(200, "text/plain", lastAccuracy);
  });

  server.begin();
}

void loop() {
  server.handleClient();
}
