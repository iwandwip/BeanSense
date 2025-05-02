
void selectAddDatasetMenu() {
  datasetIndex = 0;
  bool inDatasetMenu = true;

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.println("Add Dataset:");
  displayDatasetChoices();

  while (inDatasetMenu) {
    button.loop();
    int CLK_state = digitalRead(CLK_PIN);

    if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
      if (digitalRead(DT_PIN) == HIGH) {
        if (datasetIndex > 0) datasetIndex--;
      } else {
        if (datasetIndex < totalDatasets) datasetIndex++;
      }
      displayDatasetChoices();
      delay(150);
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();  // Debounce

      if (datasetIndex == totalDatasets) {
        // Pilih Back
        break;
      } else {
        // Pilih Dataset
        selectedDataset = datasetFiles[datasetIndex];
        selectLabelMenu();  // Panggil menu label setelah memilih dataset

        // Refresh tampilan setelah kembali dari menu label
        tft.fillScreen(TFT_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(TFT_CYAN);
        tft.setCursor(20, 10);
        tft.println("Add Dataset:");
        displayDatasetChoices();
      }
    }


    prev_CLK_state = CLK_state;
  }
}

void displayLabelPage(int page, int labelIndex, const char* labels[], int totalLabels, int labelsPerPage) {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.println("Pilih Label:");

  for (int i = 0; i <= labelsPerPage; i++) {
    if (i == labelIndex) {
      tft.setTextColor(TFT_WHITE, TFT_BLUE);
    } else {
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
    }

    tft.setCursor(40, 50 + (i * 30));

    if (i == 0) {
      tft.print("Back");
    } else {
      int labelIdx = page * labelsPerPage + (i - 1);
      if (labelIdx < totalLabels) {
        tft.print(labels[labelIdx]);
      }
    }
  }
}
// ==================== GLOBAL DEFINITIONS ====================
const char* labels[] = {
  "aKaw-D", "aKaw-L", "aKaw-M",
  "aSem-D", "aSem-L", "aSem-M",
  "rGed-D", "rGed-L", "rGed-M",
  "rTir-D", "rTir-L", "rTir-M"
};
const int totalLabels = 12;
const int labelsPerPage = 6;
const int totalPages = (totalLabels + labelsPerPage - 1) / labelsPerPage;

// ==================== DISPLAY FUNCTION ====================
void displayLabelPage(int page, int labelIndex) {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.print("Pilih Label (");
  tft.print(page + 1);
  tft.print("/");
  tft.print(totalPages);
  tft.println(")");

  // Opsi Back
  if (labelIndex == 0) tft.setTextColor(TFT_WHITE, TFT_BLUE);
  else tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(20, 50);
  tft.print("Back");

  // Label 6 item, kolom kiri dan kanan
  for (int i = 0; i < labelsPerPage; i++) {
    int globalIdx = page * labelsPerPage + i;
    if (globalIdx >= totalLabels) break;

    // Hitung kolom dan baris: i=0..2 kiri, i=3..5 kanan
    int col = (i < 3) ? 0 : 1;
    int row = (i % 3);
    int x = 20 + col * 120;  // 20px margin + 120px per kolom
    int y = 90 + row * 40;   // mulai 90px, jarak 40px tiap baris

    int displayIndex = i + 1;  // 1..6, karena 0=Back
    if (labelIndex == displayIndex) {
      tft.setTextColor(TFT_WHITE, TFT_BLUE);
    } else {
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
    }
    tft.setCursor(x, y);
    tft.print(labels[globalIdx]);
  }
}

// ==================== MAIN SELECTION FUNCTION ====================
void selectLabelMenu() {
  int page = 0;
  int labelIndex = 0;  // 0=Back, 1..6=label di halaman
  bool inMenu = true;

  displayLabelPage(page, labelIndex);

  while (inMenu) {
    button.loop();
    int clk = digitalRead(CLK_PIN);

    if (clk != prev_CLK_state && clk == HIGH) {
      // Putar kiri = naik, kanan = turun
      if (digitalRead(DT_PIN) == HIGH) {
        // scroll up
        if (labelIndex > 0) {
          labelIndex--;
        } else if (page > 0) {
          page--;
          labelIndex = labelsPerPage;  // pos terakhir halaman atas
        }
      } else {
        // scroll down
        if (labelIndex < labelsPerPage) {
          // cek masih ada label
          if (page * labelsPerPage + labelIndex < totalLabels) {
            labelIndex++;
          }
        } else if (page < totalPages - 1) {
          page++;
          labelIndex = 0;  // kembali ke Back
        }
      }
      displayLabelPage(page, labelIndex);
      delay(300);
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();  // debounce

      if (labelIndex == 0) {
        // Back
        if (page == 0) {
          inMenu = false;  // keluar seluruh menu
        } else {
          page--;
          labelIndex = 0;
          displayLabelPage(page, labelIndex);
        }
      } else {
        // Pilih label
        int selIdx = page * labelsPerPage + (labelIndex - 1);
        if (selIdx < totalLabels) {
          String chosen = labels[selIdx];
          // tft.fillScreen(TFT_BLACK);
          // tft.setCursor(20, 60);
          // tft.setTextColor(TFT_GREEN);
          // tft.print("Dipilih: ");
          // tft.println(chosen);
          // delay(1500);
          inMenu = false;

          sensorMenu(chosen);  // <=== Tambahin ini
        }
      }
    }

    prev_CLK_state = clk;
  }
}





// ==================== SENSOR MENU FUNCTION ====================


void sensorMenu(String label) {
  const int menuCount = 6;
  int menuIndex = 0;
  bool inMenu = true;

  tft.fillScreen(TFT_BLACK);

  int lastMenuIndex = -1;
  int lastCLK = digitalRead(CLK_PIN);

  while (inMenu) {
    button.loop();

    String menuTitles[menuCount] = {
      "Inhale", "Save[" + String(countDataLoad) + "]", "Next",
      "Flush", "Retry", "Back"
    };
    int clk = digitalRead(CLK_PIN);

    drawSpiderChart();  // Gambar radar chart
    // Redraw radar chart dan menu hanya saat menuIndex berubah
    if (menuIndex != lastMenuIndex) {
      tft.fillRect(0, 0, 240, 180, TFT_BLACK);  // clear area radar chart

      readLogic = true;
      String dataStr = "";
      // Gambar label di kanan spiderchart
      tft.setTextSize(2);
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
      // Set warna
      uint16_t boxColor = TFT_WHITE;   // warna kotak
      uint16_t textColor = TFT_BLACK;  // warna teks

      // Koordinat dan ukuran box
      int boxX = 225;
      int boxY = 10;
      int boxWidth = 85;
      int boxHeight = 35;


      // Gambar kotak
      tft.fillRect(boxX, boxY, boxWidth, boxHeight, boxColor);

      // Atur warna teks
      tft.setTextColor(textColor);

      // Atur posisi teks agar lebih rapi di dalam box
      // Misal offset sedikit dari pojok kiri atas
      tft.setCursor(boxX + 5, boxY + (boxHeight / 2) - 7);  // 5px dari kiri, center vertikal kira-kira

      // Tulis label
      tft.print(label);
      // --- Gambar tabel di bawah box label ---
      int tableX = boxX + 25;
      int tableY = boxY + boxHeight + 5;
      int tableWidth = 40;
      int tableHeight = 120;
      int rowHeight = tableHeight / 6;  // 1 baris header + 5 data

      // Gambar kotak tabel luar
      tft.drawRect(tableX, tableY, tableWidth, tableHeight, TFT_WHITE);

      // Header
      tft.setTextSize(1);
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
      tft.setCursor(tableX + 5, tableY + 5);
      tft.print("No |");

      // Garis horizontal bawah header
      tft.drawFastHLine(tableX, tableY + rowHeight, tableWidth, TFT_WHITE);

      // Garis vertikal pemisah "No" dan "Data"
      tft.drawFastVLine(tableX + 20, tableY, tableHeight, TFT_WHITE);

      // Isi Data
      for (int i = 0; i < 5; i++) {
        int rowY = tableY + rowHeight * (i + 1);  // Baris ke 1..5

        // Nomor
        tft.setCursor(tableX + 5, rowY + 5);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        tft.print(i + 1);

        // Jika ada data, tampilkan kotak hijau + huruf 'v'
        if (tempData[i] != "") {
          // Gambar kotak hijau kecil
          int boxSize = 10;
          int boxX = tableX + 25;
          int boxY = rowY + 3;

          tft.fillRect(boxX, boxY, boxSize, boxSize, TFT_GREEN);
          tft.setTextColor(TFT_BLACK, TFT_GREEN);
          tft.setCursor(boxX + 2, boxY + 1);  // sedikit offset biar huruf v di tengah
          tft.print("v");
        }
      }

      // --- Gambar Menu 2x3 dengan Box Per Tombol ---
      int startX = 0;
      int startY = 180;  // Area bawah
      int buttonHeight = 25;
      int spacingX = 10;
      int spacingY = 10;

      for (int i = 0; i < menuCount; i++) {
        int buttonWidth = 80;  // Default
        int col = i % 3;
        int row = i / 3;
        int x = startX + col * (buttonWidth + spacingX);
        int y = startY + row * (buttonHeight + spacingY);

        // Menyesuaikan panjang tombol berdasarkan teks
        if (menuTitles[i] == "Inhale" || menuTitles[i] == "Flush") {
          buttonWidth = 130;  // Panjang tambah 40px
        } else if (menuTitles[i] == "Save" || menuTitles[i] == "Retry") {
          buttonWidth = 100;  // Panjang sama untuk Save dan Retry
        } else if (menuTitles[i] == "Next" || menuTitles[i] == "Back") {
          buttonWidth = 70;  // Panjang kurang 10px
        }

        // Penyesuaian posisi Save, Retry, Next, Back
        if (row == 0 && (i == 1)) x += 50;  // Save agak ke kanan 40px
        if (row == 1 && (i == 4)) x += 50;  // Retry agak ke kanan 40px
        if (i == 2 || i == 5) x += 60;      // Next & Back agak lebih kanan 50px

        // Gambar kotak pembungkus tiap tombol dengan panjang yang disesuaikan
        tft.drawRect(x - 2, y - 2, buttonWidth + 4, buttonHeight + 4, TFT_WHITE);  // box di sekitar tombol

        // Highlight tombol aktif
        if (i == menuIndex) {
          tft.fillRect(x, y, buttonWidth - 2, buttonHeight, TFT_BLUE);
          tft.setTextColor(TFT_WHITE, TFT_BLUE);
        } else {
          tft.fillRect(x, y, buttonWidth - 2, buttonHeight, TFT_BLACK);
          tft.setTextColor(TFT_WHITE, TFT_BLACK);
        }

        tft.setCursor(x + 5, y + 5);
        tft.setTextSize(2);
        if (menuTitles[i] == "Inhale") {
          tft.print("Inhale:");
          tft.print(pompaNoseLogic ? "ON" : "OFF");
        } else if (menuTitles[i] == "Flush") {
          tft.print("Flush:");
          tft.print(pompaFlushLogic ? "ON" : "OFF");
        } else {
          tft.print(menuTitles[i]);
        }
      }
      // --- Akhir Gambar Menu ---

      lastMenuIndex = menuIndex;
    }

    // Perbaikan stabil pembacaan encoder
    if (clk != lastCLK) {
      if (millis() % 100 > 3) {
        if (clk == HIGH) {
          if (digitalRead(DT_PIN) == LOW) {
            menuIndex++;
            if (menuIndex >= menuCount) menuIndex = 0;  // wrap around
          } else {
            menuIndex--;
            if (menuIndex < 0) menuIndex = menuCount - 1;  // wrap around
          }
        }
        lastCLK = clk;
      }
    }

    // Jika tombol ditekan
    if (button.isPressed()) {
      while (button.isPressed()) button.loop();  // debounce tombol


      tft.setCursor(20, 250);
      tft.setTextSize(2);
      tft.setTextColor(TFT_GREEN, TFT_BLACK);

      switch (menuIndex) {
        case 0:  // Inhale toggle
          pompaNoseLogic = !pompaNoseLogic;
          digitalWrite(pumpInhale, !pompaNoseLogic);
          break;

        case 1:  // Save Data
          if (countDataLoad < 5) {
            String dataStr;
            for (int i = 0; i < NUM_SENSOR[datasetIndex]; i++) {
              dataStr += String(sensorValues[i]);
              if (i < NUM_SENSOR[datasetIndex] - 1) dataStr += ",";
            }
            tempData[countDataLoad] = dataStr;
            countDataLoad++;
          }
          break;

        case 2:  // Next
          inMenu = true;
          readLogic = false;
          for (int i = 0; i < 5; i++) {
            tempData[i] = label + "," + tempData[i];
          }
          saveToCsv(tempData, countDataLoad);
          for (int i = 0; i < 5; i++) tempData[i] = "";
          break;

        case 3:  // Flush toggle
          pompaFlushLogic = !pompaFlushLogic;
          digitalWrite(pumpFlush, !pompaFlushLogic);

          break;

        case 4:  // Retry
          countDataLoad = 0;
          for (int i = 0; i < 5; i++) tempData[i] = "";

          // Clear area tabel
          tft.fillRect(225, 50, 80, 120, TFT_BLACK);

          lastMenuIndex = -1;  // paksa redraw tabel
          break;

        case 5:            // Back
          inMenu = false;  // keluar dari menu sensor
          readLogic = false;
          digitalWrite(pumpFlush, HIGH);
          digitalWrite(pumpInhale, HIGH);

          delay(500);
          break;
      }
      lastMenuIndex = -1;  // paksa redraw menu setelah aksi
    }
  }
}

void saveToCsv(String datasetPacket[], int totalData) {
  if (!SPIFFS.begin(true)) {
    Serial.println("Gagal mount SPIFFS");

    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.setTextSize(2);
    tft.setCursor(10, 10);
    tft.println("Gagal mount SPIFFS");
    return;
  }

  const char* datasetName[3] = { "/dataset4.csv", "/dataset6.csv", "/dataset8.csv" };

  fs::File file = SPIFFS.open(datasetName[datasetIndex], FILE_APPEND);
  if (!file) {
    Serial.println("Gagal membuka file untuk menulis (append)");

    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_RED, TFT_BLACK);
    tft.setTextSize(2);
    tft.setCursor(10, 10);
    tft.println("Gagal menyimpan!");
    return;
  }

  // Menulis data ke file
  for (int i = 0; i < totalData; i++) {
    file.println(datasetPacket[i]);
  }
  file.close();

  Serial.println("Selesai memperbarui file (append)");

  // Bersihkan layar TFT dan tampilkan data
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(1);
  tft.setCursor(10, 10);

  tft.print("Data Tersimpan : ");
  tft.println(datasetName[datasetIndex]);  // Menampilkan nama file

  // Menampilkan totalData isi dataset
  for (int i = 0; i < totalData; i++) {
    tft.setCursor(10, 50 + i * 20);  // Atur posisi supaya rapi
    tft.println(datasetPacket[i]);
  }
  while (1) {
    button.loop();
    if (button.isPressed()) break;
  }
}
