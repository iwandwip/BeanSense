//===================display=====================

void setupDisplay() {
  //tft display
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);


  pinMode(pumpInhale, OUTPUT);
  pinMode(pumpFlush, OUTPUT);
  digitalWrite(pumpFlush, HIGH);
  digitalWrite(pumpInhale, HIGH);

  // rotary encoder
  pinMode(CLK_PIN, INPUT);
  pinMode(DT_PIN, INPUT);
  button.setDebounceTime(50);
  prev_CLK_state = digitalRead(CLK_PIN);
}


void loadingBar(int duration) {
  tft.fillScreen(TFT_BLACK);

  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextDatum(MC_DATUM);
  tft.setCursor(65, 80);
  tft.setTextSize(1);
  tft.print("Loading :");

  tft.drawRect(barX - 1, barY - 1, barWidthMax + 2, barHeight + 2, TFT_WHITE);  // border

  unsigned long startTime = millis();

  while (millis() - startTime < duration) {
    float progress = float(millis() - startTime) / duration;
    int barWidth = progress * barWidthMax;

    tft.fillRect(barX, barY, barWidth, barHeight, TFT_GREEN);
    delay(30);
  }

  // Tampilan setelah selesai
  tft.fillScreen(TFT_BLACK);
}



// ==================== FUNCTION DRAW SPIDER CHART ====================
void drawSpiderChart() {

  tft.setTextSize(1);
  int jumlahSensor = NUM_SENSOR[datasetIndex];
  float angleStep = 2 * PI / jumlahSensor;

  // Draw background grid (4 layer lingkaran)
  for (int i = 1; i <= 4; i++) {
    int level = i * RADIUS / 4;
    for (int j = 0; j < jumlahSensor; j++) {
      float angle1 = j * angleStep;
      float angle2 = ((j + 1) % jumlahSensor) * angleStep;
      int x1 = CENTER_X + cos(angle1) * level;
      int y1 = CENTER_Y + sin(angle1) * level;
      int x2 = CENTER_X + cos(angle2) * level;
      int y2 = CENTER_Y + sin(angle2) * level;
      tft.drawLine(x1, y1, x2, y2, TFT_DARKGREY);
    }
  }

  // Draw axes
  for (int i = 0; i < jumlahSensor; i++) {
    float angle = i * angleStep;
    int x = CENTER_X + cos(angle) * RADIUS;
    int y = CENTER_Y + sin(angle) * RADIUS;
    tft.drawLine(CENTER_X, CENTER_Y, x, y, TFT_WHITE);
  }

  // ========== Tambahkan Bagian Ini: Menggambar Grafik Radar ==========

  int pointX[jumlahSensor];
  int pointY[jumlahSensor];

  for (int i = 0; i < jumlahSensor; i++) {
    float angle = i * angleStep;
    float normValue = sensorValues[i] / 8000.0;  // Normalisasi terhadap 200
    int r = RADIUS * normValue;                  // Jarak dari pusat
    if (r < 10) r = 10;
    pointX[i] = CENTER_X + cos(angle) * r;
    pointY[i] = CENTER_Y + sin(angle) * r;
  }

  // Gambar garis menghubungkan semua titik
  for (int i = 0; i < jumlahSensor; i++) {
    int next = (i + 1) % jumlahSensor;  // wrap ke 0 di akhir
    tft.drawLine(pointX[i], pointY[i], pointX[next], pointY[next], TFT_BLUE);
  }

  // Opsional: Gambar bulatan kecil di setiap titik
  for (int i = 0; i < jumlahSensor; i++) {
    tft.fillCircle(pointX[i], pointY[i], 3, TFT_RED);
  }



  for (int i = 0; i < jumlahSensor; i++) {
    float angle = i * angleStep;

    int baseX, baseY;
    if (jumlahSensor == 4) {
      baseX = CENTER_X + cos(angle) * (RADIUS + 10);
      baseY = CENTER_Y + sin(angle) * (RADIUS + 10);
    } else if (jumlahSensor == 8) {
      baseX = CENTER_X + cos(angle) * (RADIUS + 10);
      baseY = CENTER_Y + sin(angle) * (RADIUS + 10);
    } else {
      baseX = CENTER_X + cos(angle) * (RADIUS + 20);
      baseY = CENTER_Y + sin(angle) * (RADIUS + 20);
    }

    int labelX, labelY;

    if (jumlahSensor == 4) {
      labelX = CENTER_X + cos(angle) * (RADIUS + 20);
      labelY = CENTER_Y + sin(angle) * (RADIUS + 20);
    } else if (jumlahSensor == 8) {
      labelX = CENTER_X + cos(angle) * (RADIUS + 25);
      labelY = CENTER_Y + sin(angle) * (RADIUS + 25);
    } else {
      labelX = CENTER_X + cos(angle) * (RADIUS + 35);
      labelY = CENTER_Y + sin(angle) * (RADIUS + 35);
    }
    // Default posisi
    int valueX = baseX - 10;
    int valueY = baseY - 5;
    int adjLabelX = labelX - 10;
    int adjLabelY = labelY - 5;

    // Cek apakah sudut dekat ke kanan (0°) atau kiri (180°)
    if ((angle <= PI / 8) || (angle >= 15 * PI / 8) || (angle >= 7 * PI / 8 && angle <= 9 * PI / 8)) {
      // kanan (0°) atau kiri (180°)
      valueY = baseY + 5;       // value turun ke bawah
      adjLabelY = labelY - 15;  // label naik ke atas
    }

    // Tulis label sensor
    tft.setTextColor(TFT_BLUE, TFT_BLACK);
    tft.setCursor(adjLabelX, adjLabelY);
    tft.print(sensorLabels[i]);

    // Tulis value sensor
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setCursor(valueX, valueY);
    tft.print(sensorValues[i]);
  }
}


void displayGridSaved(int sensorTempValues[]) {
  tft.setCursor(220, 15);
    tft.print("record data: ");
  for (int i = 0; i < NUM_SENSOR[datasetIndex]; i++) {
    tft.setCursor(220 , (i * 15) + 30);
    tft.print(String(sensorLabels[i]) + " = " + String(sensorTempValues[i]));
  }
  int jumlahSensor = NUM_SENSOR[datasetIndex];
  float angleStep = 2 * PI / jumlahSensor;
  int pointX[jumlahSensor];
  int pointY[jumlahSensor];

  for (int i = 0; i < jumlahSensor; i++) {
    float angle = i * angleStep;
    float normValue = sensorTempValues[i] / 8000.0;  // Normalisasi terhadap 200
    int r = RADIUS * normValue;                  // Jarak dari pusat
    if (r < 10) r = 10;
    pointX[i] = CENTER_X + cos(angle) * r;
    pointY[i] = CENTER_Y + sin(angle) * r;
  }

  // Gambar garis menghubungkan semua titik
  for (int i = 0; i < jumlahSensor; i++) {
    int next = (i + 1) % jumlahSensor;  // wrap ke 0 di akhir
    tft.drawLine(pointX[i], pointY[i], pointX[next], pointY[next], TFT_BLUE);
  }

  // Opsional: Gambar bulatan kecil di setiap titik
  for (int i = 0; i < jumlahSensor; i++) {
    tft.fillCircle(pointX[i], pointY[i], 3, TFT_RED);
  }
}
