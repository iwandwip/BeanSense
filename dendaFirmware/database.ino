

// Fungsi command dengan blocking until result received
void commandTrain(String cmd, String method, String dataset) {
  logicServer = true;
  delay(1000);
  lastAccuracy = "idle";  // Reset akurasi
  commandToSend = cmd;
  commandReady = true;
  Serial.println("📤 Command sent: " + cmd);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE,TFT_BLACK);
  tft.setCursor(10, 15);
  tft.print("Process Training Dataset");
  tft.setCursor(10, 40);
  tft.print("Dataset : " + dataset);
  tft.setCursor(10, 65);
  tft.print("Method : " + method);
  tft.setCursor(10, 115);
  tft.print("Waiting Result...");
  // Tunggu hingga result != "idle"
  static uint32_t start = millis(), elapse;
  while (lastAccuracy == "idle") {
    elapse = millis() - start;
    if (millis() % 100 > 2) {
      tft.setCursor(10, 90);
      tft.print("Duration : " + String(elapse) + " ms ");
    }
  }

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  // Cetak waktu durasi

  tft.setCursor(30, 15);
  tft.print("Train Dataset Result");
  tft.setCursor(10, 40);
  tft.print("Dataset : " + dataset);
  tft.setCursor(10, 65);
  tft.print("Method : " + method);

  // Parsing lastAccuracy
  String parts[4];  // Akan menampung 4 bagian: avg_accuracy, avg_f1Score, avg_auc, memory_used
  int idx = 0;

  int fromIndex = 0;
  while (idx < 4) {
    int commaIndex = lastAccuracy.indexOf(',', fromIndex);
    if (commaIndex == -1) {
      parts[idx++] = lastAccuracy.substring(fromIndex);  // bagian terakhir
      break;
    }
    parts[idx++] = lastAccuracy.substring(fromIndex, commaIndex);
    fromIndex = commaIndex + 1;
  }

  // Cetak hasil parsing
  tft.setCursor(10, 90);
  tft.print("Duration: " + String((float)elapse / 1000.f) + " s");
  tft.setCursor(10, 115);
  tft.print(parts[0]);  // avg_accuracy
  tft.setCursor(10, 140);
  tft.print(parts[1]);  // avg_f1Score
  tft.setCursor(10, 165);
  tft.print(parts[2]);  // avg_auc
  tft.setCursor(10, 190);
  tft.print(parts[3]);  // memory_used

  logicServer = false;
  while (digitalRead(27))
    ;
  displayMethodPage(0 , 0 );
  
}

// Fungsi command dengan blocking until result received
void commandPredict(String cmd, String method, String dataset) {
  logicServer = true;
  delay(1000);
  lastAccuracy = "idle";  // Reset akurasi
  commandToSend = cmd  ;
  commandReady = true;
  Serial.println("📤 Command sent: " + cmd);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE,TFT_BLACK);
  tft.setCursor(10, 15);
  tft.print("Process Predicting Data");
  tft.setCursor(10, 40);
  tft.print("Dataset : " + dataset);
  tft.setCursor(10, 65);
  tft.print("Method : " + method);
  tft.setCursor(10, 115);
  tft.print("Waiting Result...");
  // Tunggu hingga result != "idle"
  static uint32_t start = millis(), elapse;
  while (lastAccuracy == "idle") {
    elapse = millis() - start;
    if (millis() % 100 > 2) {
      tft.setCursor(10, 90);
      tft.print("Duration : " + String(elapse) + " ms ");
    }
  }

  // Parsing lastAccuracy
  String parts[4];  // Akan menampung 5 bagian: avg_accuracy, avg_f1Score, avg_auc, memory_used , label
  int idx = 0;

  int fromIndex = 0;
  while (idx < 5) {
    int commaIndex = lastAccuracy.indexOf(',', fromIndex);
    if (commaIndex == -1) {
      parts[idx++] = lastAccuracy.substring(fromIndex);  // bagian terakhir
      break;
    }
    parts[idx++] = lastAccuracy.substring(fromIndex, commaIndex);
    fromIndex = commaIndex + 1;
  }

  // Cetak hasil parsing

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  // Cetak waktu durasi

  tft.setCursor(20, 15);
  tft.print("Method : " + method);
  tft.setCursor(10, 40);
  tft.print("Dataset : " + dataset);
  tft.setCursor(10, 65);
  tft.print(parts[4]);

  tft.setCursor(10, 90);
  tft.print("Duration: " + String((float)elapse / 1000.f) + " s");
  tft.setCursor(10, 115);
  tft.print(parts[0]);  // avg_accuracy
  tft.setCursor(10, 140);
  tft.print(parts[1]);  // avg_f1Score
  tft.setCursor(10, 165);
  tft.print(parts[2]);  // avg_auc
  tft.setCursor(10, 190);
  tft.print(parts[3]);  // memory_used


  logicServer = false;
  while (digitalRead(27))
    ;
  displayMethodPage(0 , 0 );
}
// Fungsi membaca file CSV dari SPIFFS
String readCSV(const char* filename) {
  fs::File file = SPIFFS.open(filename, "r");
  if (!file) return "Failed to open file";
  String fileContent = file.readString();
  file.close();
  return fileContent;
}


void setupServer() {
  // Setup SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("Failed to mount SPIFFS");
    return;
  }

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

void serverHandler(bool logicServer) {
  if (logicServer) {
    server.handleClient();
  }
}

void setupFileSystem() {
  if (!SPIFFS.begin(true)) {
    tft.setCursor(20, 50);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.println("SPIFFS GAGAL!");
    while (1)
      ;
  }
}
