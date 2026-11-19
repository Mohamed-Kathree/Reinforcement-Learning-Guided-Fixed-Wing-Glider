#include <WiFi.h>
#include <esp_now.h>

static const uint8_t FLIGHT_MAC[6] = {0xF8,0xB3,0xB7,0x30,0x33,0x40};

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5,5,0)
  #define SEND_CB_ARG const wifi_tx_info_t*
#else
  #define SEND_CB_ARG const uint8_t*
#endif

void onRecv(const esp_now_recv_info_t*, const uint8_t* d, int len) {
  for (int i = 0; i < len; i++) Serial.write(d[i]);
}

void onSent(SEND_CB_ARG, esp_now_send_status_t s) {
  if (s != ESP_NOW_SEND_SUCCESS) Serial.println("\n  [tx to flight failed]");
}

void setup() {
  Serial.begin(115200);
  delay(400);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    while (1) delay(1000);
  }

  esp_now_peer_info_t p = {};
  memcpy(p.peer_addr, FLIGHT_MAC, 6);
  p.channel = 0;
  p.encrypt = false;
  if (esp_now_add_peer(&p) != ESP_OK) {
    Serial.println("add_peer failed");
    while (1) delay(1000);
  }

  esp_now_register_recv_cb(onRecv);
  esp_now_register_send_cb(onSent);

  Serial.println();
  Serial.printf("=== ground station up, my MAC %s ===\n", WiFi.macAddress().c_str());
  Serial.println("=== expecting flight board f8:b3:b7:30:33:40 ===");
}

void loop() {
  if (Serial.available()) {
    uint8_t c = Serial.read();
    esp_now_send(FLIGHT_MAC, &c, 1);
  }
}