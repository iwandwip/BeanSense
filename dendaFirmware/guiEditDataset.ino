// =================== CONSTANT ===================
const int BACK_OPTION = -1;

// =================== LOOP ===================
void loop() {
  button.loop();
  guiHome(); // Main entry point for all menu functionality
}

// =================== MAIN MENU SYSTEM ===================
void guiHome() {
  static bool inMainMenu = true;

  if (inMainMenu) {
    displayHome();
    inMainMenu = false;
  }

  int CLK_state = digitalRead(CLK_PIN);
  if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
    if (digitalRead(DT_PIN) == HIGH) {
      if (menuIndex > 0) menuIndex--;
    } else {
      if (menuIndex < totalMenuItems - 1) menuIndex++;
    }
    displayHome();
    delay(100);
  }

  if (button.isPressed()) {
    while (button.isPressed()) button.loop(); // Debounce

    switch (menuIndex) {
      case 0:
        // First menu item
        break;
      case 1:
        // Second menu item
        break;
      case 2:
        selectDatasetMenu(); // <-- Menu Dataset
        displayHome(); // Refresh home setelah keluar
        break;
    }
  }

  prev_CLK_state = CLK_state;
}

// =================== DISPLAY MAIN MENU ===================
void displayHome() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
  tft.setTextSize(2);
  tft.setCursor(90, 10);
  tft.println("MENU");

  for (int i = 0; i < totalMenuItems; i++) {
    if (i == menuIndex) {
      tft.setTextColor(TFT_WHITE, TFT_BLUE);
    } else {
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
    }
    tft.setCursor(40, 50 + (i * 30));
    tft.print(menuItems[i]);
  }
}

// =================== MENU PILIH DATASET ===================
void selectDatasetMenu() {
  datasetIndex = 0;
  bool inDatasetMenu = true;

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.println("Pilih Dataset:");
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
      while (button.isPressed()) button.loop(); // Debounce

      if (datasetIndex == totalDatasets) {
        // Pilih Back
        inDatasetMenu = false;
      } else {
        // Pilih Dataset
        selectedDataset = datasetFiles[datasetIndex];
        viewDatasetPage();  // Masuk viewDatasetPage()

        // Habis balik dari viewDatasetPage, refresh pilih dataset
        tft.fillScreen(TFT_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(TFT_CYAN);
        tft.setCursor(20, 10);
        tft.println("Pilih Dataset:");
        displayDatasetChoices();
      }
    }

    prev_CLK_state = CLK_state;
  }
}

// =================== DISPLAY PILIHAN DATASET ===================
void displayDatasetChoices() {
  for (int i = 0; i <= totalDatasets; i++) {
    if (i == datasetIndex) {
      tft.setTextColor(TFT_WHITE, TFT_BLUE);
    } else {
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
    }
    tft.setCursor(40, 50 + (i * 30));

    if (i == totalDatasets) {
      tft.print("Back");
    } else {
      tft.print(datasetFiles[i]);
    }
  }
}

// =================== TAMPILKAN ISI DATASET ===================
void viewDatasetPage() {
  scrollOffset = 0;
  selectedLine = 0; // 0 = Back
  totalLines = 0;
  const int linesPerPage = 10;
  bool inDatasetPage = true;

  // Hitung jumlah baris
  fs::File file = SPIFFS.open(selectedDataset, "r");
  if (!file) {
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_RED);
    tft.setCursor(10, 60);
    tft.println("Gagal buka file:");
    tft.println(selectedDataset);
    delay(2000);
    return;
  }

  file.readStringUntil('\n'); // Skip header
  while (file.available()) {
    String line = file.readStringUntil('\n');
    if (line.length() > 1) totalLines++;
  }
  file.close();

  if (totalLines == 0) {
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(10, 60);
    tft.setTextColor(TFT_RED);
    tft.println("Dataset kosong.");
    delay(2000);
    return;
  }

  displayDatasetContent();
  displayCursor();

  while (inDatasetPage) {
    button.loop();
    int CLK_state = digitalRead(CLK_PIN);
    if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
      if (digitalRead(DT_PIN) == HIGH) {
        // Scroll Up
        if (selectedLine > 0) {
          selectedLine--;
        } else if (scrollOffset > 0) {
          scrollOffset -= linesPerPage;
          selectedLine = linesPerPage;
          if (scrollOffset + selectedLine - 1 >= totalLines) {
            selectedLine = (totalLines - scrollOffset);
          }
          displayDatasetContent();
        } else {
          // Tambahan: dari atas, lompat ke halaman terakhir
          int lastPageOffset = (totalLines / linesPerPage) * linesPerPage;
          if (lastPageOffset >= totalLines) {
            lastPageOffset -= linesPerPage;
          }
          scrollOffset = lastPageOffset;
          selectedLine = totalLines - scrollOffset;
          displayDatasetContent();
        }
      } else {
        // Scroll Down
        int maxVisibleLines = min(linesPerPage, totalLines - scrollOffset);
        if (selectedLine < maxVisibleLines) {
          selectedLine++;
        } else if (scrollOffset + linesPerPage < totalLines) {
          scrollOffset += linesPerPage;
          selectedLine = 1;
          displayDatasetContent();
        } else {
          // Tambahan: dari bawah, lompat ke halaman pertama
          scrollOffset = 0;
          selectedLine = 0;
          displayDatasetContent();
        }
      }
      displayCursor();
      delay(120);
    }


    if (button.isPressed()) {
      while (button.isPressed()) button.loop();

      if (selectedLine == 0) {
        // Pilih Back
        return;
      } else {
        int selectedRow = scrollOffset + (selectedLine - 1);

        if (selectedRow >= 0 && selectedRow < totalLines) {
          editDataRow(selectedRow);

          // Setelah edit refresh tampilan
          displayDatasetContent();
          displayCursor();
        }
      }
    }

    prev_CLK_state = CLK_state;
  }
}


// =================== TAMPILKAN ISI DATA ===================
void displayDatasetContent() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(1);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(5, 5);
  tft.print(selectedDataset);
  tft.print(" - Page ");
  tft.print((scrollOffset / linesPerPage) + 1);
  tft.print("/");
  tft.print((totalLines + linesPerPage - 1) / linesPerPage);

  tft.setCursor(20, 25);
  tft.setTextColor(TFT_WHITE);
  tft.print("Back");

  fs::File file = SPIFFS.open(selectedDataset, "r");
  file.readStringUntil('\n'); // Skip header

  for (int i = 0; i < scrollOffset; i++) {
    file.readStringUntil('\n');
  }

  int visibleLines = min(linesPerPage, totalLines - scrollOffset);
  for (int i = 0; i < visibleLines; i++) {
    if (file.available()) {
      String line = file.readStringUntil('\n');
      if (line.length() > 30) line = line.substring(0, 27) + "...";

      tft.setTextColor(TFT_WHITE);
      tft.setCursor(20, 45 + (i * 20));
      tft.print(line);
    }
  }
  file.close();
}

// =================== TAMPILKAN KURSOR ===================
void displayCursor() {
  // Clear semua posisi cursor
  tft.fillRect(5, 25, 15, 15, TFT_BLACK); // Back
  for (int i = 0; i < linesPerPage; i++) {
    tft.fillRect(5, 45 + (i * 20), 15, 15, TFT_BLACK); // Data rows
  }

  // Gambar cursor segitiga
  if (selectedLine == BACK_OPTION) {
    tft.fillTriangle(
      20, 25 + 5,
      10, 25,
      10, 25 + 10,
      TFT_BLUE
    );
  } else {
    tft.fillTriangle(
      20, 45 + ((selectedLine - 1) * 20) + 5,
      10, 45 + ((selectedLine - 1) * 20),
      10, 45 + ((selectedLine - 1) * 20) + 10,
      TFT_BLUE
    );
  }
}

// =================== EDIT ROW ===================
void editDataRow(int rowNumber) {
  int submenuIndex = 0;
  const char* subMenuItems[] = {"Predict", "Delete", "Back"};
  const int subMenuCount = 3;
  bool inRowMenu = true;

  String lineData;
  fs::File file = SPIFFS.open(selectedDataset, "r");
  file.readStringUntil('\n');
  for (int i = 0; i <= rowNumber; i++) {
    if (file.available()) {
      lineData = file.readStringUntil('\n');
    }
  }
  file.close();

  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
  tft.setCursor(10, 10);
  tft.printf("Row #%d", rowNumber + 1);
  tft.setCursor(10, 35);
  tft.setTextColor(TFT_YELLOW);
  tft.print("Data: ");
  tft.println(lineData);

  for (int i = 0; i < subMenuCount; i++) {
    if (i == submenuIndex) {
      tft.setTextColor(TFT_BLACK, TFT_GREEN);
    } else {
      tft.setTextColor(TFT_WHITE, TFT_BLACK);
    }
    tft.setCursor(30, 70 + i * 30);
    tft.print(subMenuItems[i]);
  }

  while (inRowMenu) {
    button.loop();
    int CLK_state = digitalRead(CLK_PIN);

    if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
      if (digitalRead(DT_PIN) == HIGH) {
        submenuIndex = (submenuIndex + subMenuCount - 1) % subMenuCount;
      } else {
        submenuIndex = (submenuIndex + 1) % subMenuCount;
      }

      for (int i = 0; i < subMenuCount; i++) {
        if (i == submenuIndex) {
          tft.setTextColor(TFT_BLACK, TFT_GREEN);
        } else {
          tft.setTextColor(TFT_WHITE, TFT_BLACK);
        }
        tft.setCursor(30, 70 + i * 30);
        tft.print(subMenuItems[i]);
      }

      delay(150);
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();

      switch (submenuIndex) {
        case 0:
          tft.fillScreen(TFT_BLACK);
          tft.setCursor(10, 40);
          tft.setTextColor(TFT_CYAN);
          tft.print("Predicting...\nData: ");
          tft.println(lineData);
          delay(2000);
          inRowMenu = false;
          break;

        case 1:
          deleteRowFromCSV(rowNumber);
          totalLines--;
          if (selectedLine > totalLines - scrollOffset + 1) {
            selectedLine = max(0, totalLines - scrollOffset + 1);
          }
          inRowMenu = false;
          break;

        case 2:
          inRowMenu = false;
          break;
      }
    }

    prev_CLK_state = CLK_state;
  }
}

// =================== DELETE ROW ===================
void deleteRowFromCSV(int targetRow) {
  fs::File original = SPIFFS.open(selectedDataset, "r");
  fs::File temp = SPIFFS.open("/temp.csv", "w");

  if (!original || !temp) {
    tft.fillScreen(TFT_RED);
    tft.setTextColor(TFT_WHITE);
    tft.setCursor(10, 60);
    tft.println("Gagal buka file!");
    delay(2000);
    return;
  }

  int currentLine = 0;
  String header = original.readStringUntil('\n');
  temp.println(header);

  while (original.available()) {
    String line = original.readStringUntil('\n');
    if (currentLine != targetRow) {
      temp.println(line);
    }
    currentLine++;
  }

  original.close();
  temp.close();

  SPIFFS.remove(selectedDataset);
  SPIFFS.rename("/temp.csv", selectedDataset);

  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_GREEN);
  tft.setCursor(10, 80);
  tft.println("Data berhasil dihapus!");
  delay(1500);
}
