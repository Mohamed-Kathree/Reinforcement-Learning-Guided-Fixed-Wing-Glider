/*
 * ground_station.ino -- ESP-NOW <-> USB bridge for the flight computer.
 *
 * Radio -> USB: every received packet is written to USB unchanged (text lines
 * and binary telemetry frames alike).
 *
 * USB -> radio (changed 19 Nov 2026; the original sent ONE BYTE PER PACKET,
 * which split binary command frames so the flight computer treated them as
 * text):
 *   - a byte 0xA5 (RLG_SYNC) at a frame boundary starts a binary command frame;
 *     its 14 bytes (rlg_command_t) go out as ONE packet. Text never contains
 *     0xA5 (console text is ASCII).
 *   - anything else is console text, sent one line per packet (on '\n', when
 *     the buffer fills, or after 30 ms of silence).
 * A frame left incomplete for 30 ms is discarded and counted.
 */
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

static const uint8_t FLIGHT_MAC[6] = {0xF8, 0xB3, 0xB7, 0x30, 0x33, 0x40};
#define RLG_SYNC 0xA5
#define CMD_SIZE 14
#define IDLE_FLUSH_MS 30

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 5, 0)
  #define SEND_CB_ARG const wifi_tx_info_t*
#else
  #define SEND_CB_ARG const uint8_t*
#endif

static volatile uint32_t txFailed = 0;
static uint32_t partialDropped = 0;

// Received packets are copied into a queue and written to USB from loop():
// writing a whole packet to the UART inside the Wi-Fi callback blocked the
// Wi-Fi task for up to ~20 ms per packet (the suspected cause of the garbled
// characters seen on 19 Nov 2026).
struct RxPkt { uint8_t len; uint8_t d[250]; };
static QueueHandle_t rxQ;
static volatile uint32_t rxDropped = 0;

void onRecv(const esp_now_recv_info_t*, const uint8_t* d, int len) {
  if (len <= 0 || len > 250) return;
  RxPkt p; p.len = (uint8_t)len; memcpy(p.d, d, len);
  if (xQueueSend(rxQ, &p, 0) != pdTRUE) rxDropped = rxDropped + 1;
}

void onSent(SEND_CB_ARG, esp_now_send_status_t s) {
  if (s != ESP_NOW_SEND_SUCCESS) txFailed = txFailed + 1;
}

void setup() {
  rxQ = xQueueCreate(32, sizeof(RxPkt));
  Serial.setTxBufferSize(4096);
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
  uint8_t mac[6] = {0};
  esp_wifi_get_mac(WIFI_IF_STA, mac);
  Serial.println();
  Serial.printf("=== ground station up, my MAC %02X:%02X:%02X:%02X:%02X:%02X ===\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  Serial.println("=== expecting flight board F8:B3:B7:30:33:40; frames bridged whole ===");
}

static uint8_t frame[CMD_SIZE];
static uint8_t frameLen = 0;
static uint8_t text[200];
static uint8_t textLen = 0;
static uint32_t lastByteMs = 0;

static void flushText() {
  if (textLen) { esp_now_send(FLIGHT_MAC, text, textLen); textLen = 0; }
}

void loop() {
  RxPkt pkt;
  while (xQueueReceive(rxQ, &pkt, 0) == pdTRUE) Serial.write(pkt.d, pkt.len);
  uint32_t now = millis();
  while (Serial.available()) {
    uint8_t c = Serial.read();
    lastByteMs = now;
    if (frameLen > 0) {                       // inside a binary frame
      frame[frameLen++] = c;
      if (frameLen == CMD_SIZE) { esp_now_send(FLIGHT_MAC, frame, CMD_SIZE); frameLen = 0; }
    } else if (c == RLG_SYNC) {               // start of a binary frame
      flushText();
      frame[0] = c; frameLen = 1;
    } else {                                  // console text
      text[textLen++] = c;
      if (c == '\n' || textLen >= sizeof(text)) flushText();
    }
  }
  if (now - lastByteMs > IDLE_FLUSH_MS) {
    if (frameLen) { partialDropped++; frameLen = 0; }
    flushText();
  }
}
