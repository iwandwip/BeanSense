
void selectPredictDatasetMenu() {
  datasetIndex = 0;
  bool inDatasetMenu = true;

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.println("Predict Dataset:");
  displaySensorChoices();

  while (inDatasetMenu) {
    button.loop();
    int CLK_state = digitalRead(CLK_PIN);

    if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
      if (digitalRead(DT_PIN) == HIGH) {
        if (datasetIndex > 0) datasetIndex--;
      } else {
        if (datasetIndex < totalDatasets) datasetIndex++;
      }
      tft.setCursor(20, 10);
      tft.println("Predict Dataset:");

      displaySensorChoices();
      delay(300);
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();  // Debounce

      if (datasetIndex == totalDatasets) {
        // Pilih Back
        break;
      } else {
        // Pilih Dataset
        selectedDataset = datasetFiles[datasetIndex];
        takeSampleDataMenu();
      }
    }


    prev_CLK_state = CLK_state;
  }
}





// ==================== SENSOR MENU FUNCTION ====================


void takeSampleDataMenu() {
  const int menuCount = 6;
  int menuIndex = 0;
  bool inMenu = true;
  bool save = false;
  tft.fillScreen(TFT_BLACK);
  int sensorTempValues[MAX_SENSORS];
  int lastMenuIndex = -1;
  int lastCLK = digitalRead(CLK_PIN);

  while (inMenu) {
    button.loop();

    String menuTitles[menuCount] = {
      "Inhale", "Save", "Next",
      "Flush", "Retry", "Back"
    };
    int clk = digitalRead(CLK_PIN);

    drawSpiderChart();  // Gambar radar chart
    if (save) displayGridSaved(sensorTempValues);
    if (menuIndex != lastMenuIndex) {
      tft.fillRect(0, 0, 240, 180, TFT_BLACK);  // clear area radar chart

      readLogic = true;
      tft.setTextSize(2);
      tft.setTextColor(TFT_WHITE, TFT_BLACK);

      // --- Label di kanan spider chart ---
      uint16_t boxColor = TFT_WHITE;
      uint16_t textColor = TFT_BLACK;

      int boxX = 225;
      int boxY = 10;
      int boxWidth = 85;
      int boxHeight = 35;



      // --- Gambar Menu 2x3 ---
      int startX = 0;
      int startY = 180;
      int buttonHeight = 25;
      int spacingX = 10;
      int spacingY = 10;

      for (int i = 0; i < menuCount; i++) {
        int buttonWidth = 80;
        int col = i % 3;
        int row = i / 3;
        int x = startX + col * (buttonWidth + spacingX);
        int y = startY + row * (buttonHeight + spacingY);

        if (menuTitles[i] == "Inhale" || menuTitles[i] == "Flush") {
          buttonWidth = 130;
        } else if (menuTitles[i] == "Save" || menuTitles[i] == "Retry") {
          buttonWidth = 100;
        } else if (menuTitles[i] == "Next" || menuTitles[i] == "Back") {
          buttonWidth = 70;
        }

        if (row == 0 && (i == 1)) x += 50;
        if (row == 1 && (i == 4)) x += 50;
        if (i == 2 || i == 5) x += 60;

        tft.drawRect(x - 2, y - 2, buttonWidth + 4, buttonHeight + 4, TFT_WHITE);

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

      lastMenuIndex = menuIndex;
    }

    if (clk != lastCLK) {
      if (millis() % 100 > 3) {
        if (clk == HIGH) {
          if (digitalRead(DT_PIN) == LOW) {
            menuIndex++;
            if (menuIndex >= menuCount) menuIndex = 0;
          } else {
            menuIndex--;
            if (menuIndex < 0) menuIndex = menuCount - 1;
          }
        }
        lastCLK = clk;
      }
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();

      tft.setCursor(20, 250);
      tft.setTextSize(2);
      tft.setTextColor(TFT_GREEN, TFT_BLACK);

      switch (menuIndex) {
        case 0:  // inhale
          pompaNoseLogic = !pompaNoseLogic;
          digitalWrite(pumpInhale, !pompaNoseLogic);
          break;

        case 1:  // save
          save = true;
          if (save = true) {
            for (int i = 0; i < NUM_SENSOR[datasetIndex]; i++) {
              sensorTempValues[i] = sensorValues[i];
            }
          }

          break;

        case 2:  // next
          if (save = true) {
            String dataStr;
            for (int i = 0; i < NUM_SENSOR[datasetIndex]; i++) {
              dataStr += String(sensorTempValues[i]);
              if (i < NUM_SENSOR[datasetIndex] - 1) dataStr += ",";
            }
            selectMethodToPredictMenu(dataStr);
          }

          inMenu = true;
          readLogic = false;


          break;

        case 3:  // flush
          pompaFlushLogic = !pompaFlushLogic;
          digitalWrite(pumpFlush, !pompaFlushLogic);
          break;

        case 4:  // retry
          tft.fillScreen(TFT_BLACK);
          save = false;

          for (int i = 0; i < NUM_SENSOR[datasetIndex]; i++) {
            sensorTempValues[i] = 0;
          }
          // Tidak perlu lagi clear tabel
          lastMenuIndex = -1;
          break;

        case 5:  // back
          inMenu = false;
          readLogic = false;

          tft.fillScreen(TFT_BLACK);
          tft.setCursor(20, 10);
          tft.println("Predict Dataset:");
          displaySensorChoices();
          pompaFlushLogic = false;
          pompaNoseLogic = false;
          digitalWrite(pumpFlush, HIGH);
          digitalWrite(pumpInhale, HIGH);
          delay(500);
          break;
      }
      lastMenuIndex = -1;
    }
  }
}




void selectMethodToPredictMenu(String lineSensorTempValues) {
  int page = 0;
  bool inMenu = true;
  Serial.println(lineSensorTempValues);
  displayMethodPage(page, methodIndex);

  while (inMenu) {
    button.loop();
    int clk = digitalRead(CLK_PIN);
    if (clk != prev_CLK_state && clk == HIGH) {
      if (millis() % 100 > 3) {
        int methodsOnThisPage = methodsPerPage;
        if (page == methodPages - 1) {
          methodsOnThisPage = totalMethods % methodsPerPage;
          if (methodsOnThisPage == 0) methodsOnThisPage = methodsPerPage;
        }

        if (digitalRead(DT_PIN) == HIGH) {
          // scroll up
          if (methodIndex > 0) {
            methodIndex--;
          } else if (page > 0) {
            page--;
            methodIndex = methodsPerPage;
            if (page == methodPages - 1) {
              int lastPageItems = totalMethods % methodsPerPage;
              if (lastPageItems != 0) methodIndex = lastPageItems;
            }
          }
        } else {
          // scroll down
          if (methodIndex < methodsOnThisPage) {
            methodIndex++;
          } else if (page < methodPages - 1) {
            page++;
            methodIndex = 0;
          }
        }

        displayMethodPage(page, methodIndex);
      }
    }

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();

      if (methodIndex == 0) {
        if (page == 0) {
          tft.fillScreen(TFT_BLACK);
          break;
        } else {
          page--;
          methodIndex = 0;
          displayMethodPage(page, methodIndex);
        }
      } else {
        int selectedIdx = page * methodsPerPage + (methodIndex - 1);
        if (selectedIdx < totalMethods) {
          String chosenMethod = methods[selectedIdx];
          // Lakukan sesuatu dengan chosenMethod, misal:
          char buff[100];
          // = "predict" + "," + String(selectedIdx) + "," + String(datasetIndex) +"," + lineSensorTempValues;
          sprintf(buff, "predict,%d,%d,", datasetIndex , selectedIdx );
          commandPredict(buff +  lineSensorTempValues, chosenMethod, datasetFiles[datasetIndex]);
        }
      }
    }

    prev_CLK_state = clk;
  }
}
