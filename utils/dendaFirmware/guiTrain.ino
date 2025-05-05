
void selectTrainDatasetMenu() {
  datasetIndex = 0;
  bool inDatasetMenu = true;

  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_CYAN);
  tft.setCursor(20, 10);
  tft.println("Train Dataset:");
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
        selectMethodMenu();

        // Refresh tampilan setelah kembali dari menu label
        tft.fillScreen(TFT_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(TFT_CYAN);
        tft.setCursor(20, 10);
        tft.println("Train Dataset:");
        displayDatasetChoices();
      }
    }


    prev_CLK_state = CLK_state;
  }
}

void displayMethodPage(int page, int methodIndex) {
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(TFT_YELLOW);
  tft.setCursor(20, 10);
  tft.print("Pilih Metode (");
  tft.print(page + 1);
  tft.print("/");
  tft.print(methodPages);
  tft.println(")");

  // Tombol Back
  if (methodIndex == 0) tft.setTextColor(TFT_WHITE, TFT_BLUE);
  else tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(20, 50);
  tft.print("Back");

  // Hitung metode yang ditampilkan di halaman ini
  int startIdx = page * methodsPerPage;
  int endIdx = startIdx + methodsPerPage;
  if (endIdx > totalMethods) endIdx = totalMethods;

  for (int i = startIdx; i < endIdx; i++) {
    int displayIdx = i - startIdx + 1;  // +1 karena 0 = Back
    int y = 90 + (displayIdx - 1) * 40;

    if (methodIndex == displayIdx)
      tft.setTextColor(TFT_WHITE, TFT_BLUE);
    else
      tft.setTextColor(TFT_WHITE, TFT_BLACK);

    tft.setCursor(20, y);
    tft.print(methods[i]);
  }
}


void selectMethodMenu() {
  int page = 0;
  bool inMenu = true;

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
          String buff = "train," + String(datasetIndex) + "," + String(selectedIdx);
          commandTrain(buff, chosenMethod, datasetFiles[datasetIndex] );
        }
      }
    }

    prev_CLK_state = clk;
  }
}
