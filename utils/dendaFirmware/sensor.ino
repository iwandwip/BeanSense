//=================setup sensor========================
void setupSensor() {
  Serial.begin(115200);

  // Initialize DHT
  dht_sensor.begin();

  // Initialize ADS1115
  if (!ads1.begin(0x48)) {
    Serial.println("ADS1115 (0x48) tidak ditemukan!");
    while (1)
      ;
  }
  if (!ads2.begin(0x49)) {
    Serial.println("ADS1115 (0x49) tidak ditemukan!");
    while (1)
      ;
  }

  ads1.setGain(GAIN_TWOTHIRDS);
  ads2.setGain(GAIN_TWOTHIRDS);
}

// ==================== FUNCTION READ SENSOR ====================

void readSensor() {
  if (!readLogic) return;
  bacaSensorDHT();
  bacaSensorMQ();
}


void bacaSensorDHT() {
  humidity = dht_sensor.readHumidity() + 0.83;
  temperature = dht_sensor.readTemperature() - 0.8;

  if (isnan(temperature) || isnan(humidity)) {
    // Serial.println("Gagal membaca dari sensor DHT!");
  }
}


void bacaSensorMQ() {
  float sensorSum[NUM_SENSORS] = { 0 };
  for (int i = 0; i < NUM_READINGS; i++) {
    sensorSum[0] += (ads1.readADC_SingleEnded(0) * ((0.0174 * temperature) + 0.932)) + (humidity - 33) * 0.173;    //MQ135
    sensorSum[1] += (ads1.readADC_SingleEnded(3) * ((0.0034 * temperature) + 1.032)) + (humidity - 33) * 0.00173;  //MQ2
    sensorSum[2] += (ads2.readADC_SingleEnded(0) * ((0.0154 * temperature) + 1.082)) + (humidity - 33) * 0.0173;   //MQ3
    sensorSum[3] += (ads2.readADC_SingleEnded(1) * ((0.0054 * temperature) + 0.932)) + (humidity - 33) * 0.00173;  //MQ6
    sensorSum[4] += (ads1.readADC_SingleEnded(1) * ((0.0094 * temperature) + 1.032)) + (humidity - 33) * 0.173;    //MQ138
    sensorSum[5] += (ads2.readADC_SingleEnded(3) * ((0.0094 * temperature) + 0.582)) + (humidity - 33) * 0.00173;  //MQ7
    sensorSum[6] += (ads1.readADC_SingleEnded(2) * ((0.0204 * temperature) + 1.332)) + (humidity - 33) * 0.773;    //MQ136
    sensorSum[7] += (ads2.readADC_SingleEnded(2) * ((0.0034 * temperature) + 0.532)) + (humidity - 33) * 0.00173;  //MQ5
  }

  sensorValues[0] = (sensorSum[0] / (NUM_READINGS + 1)) - 6000;  //135
  sensorValues[1] = (sensorSum[3] / (NUM_READINGS + 1)) + 1000;  //2
  sensorValues[2] = (sensorSum[4] / (NUM_READINGS + 1)) - 9000;  //3
  sensorValues[3] = (sensorSum[5] / (NUM_READINGS + 1));         //6
  sensorValues[4] = (sensorSum[1] / (NUM_READINGS + 1)) - 6500;  //138
  sensorValues[5] = (sensorSum[7] / (NUM_READINGS + 1)) - 3000;  //7
  sensorValues[6] = (sensorSum[2] / (NUM_READINGS + 1)) - 8000;  //136
  sensorValues[7] = (sensorSum[6] / (NUM_READINGS + 1)) - 6500;  //5
}
