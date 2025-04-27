#include <TFT_eSPI.h>
#include <ezButton.h>


// =============== PIN ===============
#define CLK_PIN 25
#define DT_PIN  26
#define SW_PIN  27

// =============== CONSTRUCTOR  ===============
TFT_eSPI tft = TFT_eSPI();
ezButton button(SW_PIN);

bool prev_CLK_state ;

void setupDisplay(){
  //tft display
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextSize(2);


  // rotary encoder
  pinMode(CLK_PIN, INPUT);
  pinMode(DT_PIN, INPUT);
  button.setDebounceTime(50);
  prev_CLK_state = digitalRead(CLK_PIN);
}