

// =================== MAIN MENU SYSTEM ===================
void guiHome() {
  static bool inMainMenu = true;
  tft.fillScreen(TFT_BLACK);
  displayHome();

  while (inMainMenu) {
    button.loop();

    int CLK_state = digitalRead(CLK_PIN);
    if (CLK_state != prev_CLK_state && CLK_state == HIGH) {
      if (digitalRead(DT_PIN) == HIGH) {
        if (menuIndex > 0) menuIndex--;
      } else {
        if (menuIndex < totalMenuItems - 1) menuIndex++;
      }
      displayHome();
      delay(300);  // debounce rotary encoder
    }
    prev_CLK_state = CLK_state;

    if (button.isPressed()) {
      while (button.isPressed()) button.loop();  // Debounce

      switch (menuIndex) {
        case 0:  // training data
          loadingBar(1500);
          selectTrainDatasetMenu();
          break;
        case 1:  // predict data
          loadingBar(1500);
          selectPredictDatasetMenu();
          break;
        case 2:  // edit dataset
          loadingBar(1500);
          selectDatasetMenu();
          break;
        case 3:  // add dataset
          loadingBar(1500);
          selectAddDatasetMenu();
          break;
      }
      // Optional: redisplay menu after submenu
      tft.fillScreen(TFT_BLACK);
      displayHome();
    }
  }
}

// =================== DISPLAY MAIN MENU ===================
void displayHome() {
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
