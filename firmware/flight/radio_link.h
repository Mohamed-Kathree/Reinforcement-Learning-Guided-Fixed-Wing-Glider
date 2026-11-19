/*
 * radio_link.h -- ESP-NOW debug link to the ground station.
 *
 * Keeps the bench sketch's RadioLink pattern (a Print subclass: text written
 * to it goes to USB Serial AND, line by line, to the ground station), with two
 * changes the flight brief requires:
 *   - esp_now_send() is NEVER called from the 200 Hz control loop. write()
 *     only queues a line; radioTask (core 0) sends and then waits
 *     RADIO_SETTLE_MS (15 ms; tuned, do not reduce).
 *   - Received ESP-NOW packets that are exactly one well-formed command frame
 *     length starting with RLG_SYNC go to the command queue (drained by the
 *     control loop into rlg_safety); anything else is console text.
 *
 * Nothing here affects flight: in FLIGHT the ESP-NOW link is monitoring only
 * and has no timeout (rlg_safety.h: two links, two separate timeouts).
 *
 * Text written from more than one task is serialised by a mutex.
 */
#pragma once
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "config.h"
#include "src/protocol/rlg_protocol.h"

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 5, 0)
  #define SEND_CB_ARG const wifi_tx_info_t*
#else
  #define SEND_CB_ARG const uint8_t*
#endif

struct RadioMsg { uint8_t len; uint8_t data[200]; };
struct CmdFrame { uint8_t data[RLG_COMMAND_SIZE]; uint8_t src; };   // src 0 radio, 1 console
static_assert(sizeof(rlg_command_t) == RLG_COMMAND_SIZE, "command size");
static_assert(sizeof(rlg_telemetry_t) == RLG_TELEMETRY_SIZE, "telemetry size");

class RadioLink : public Print {
  char tx[200];
  size_t txn = 0;
  volatile char rx[128];
  volatile uint8_t head = 0, tail = 0;
  SemaphoreHandle_t lock = nullptr;
  static RadioLink* self;

  static void onRecv(const esp_now_recv_info_t*, const uint8_t* d, int len) {
    if (!self) return;
    if (len == (int)RLG_COMMAND_SIZE && d[0] == RLG_SYNC && self->cmdQueue) {
      CmdFrame f; memcpy(f.data, d, RLG_COMMAND_SIZE); f.src = 0;
      if (xQueueSend(self->cmdQueue, &f, 0) != pdTRUE) self->cmdDropped++;
      return;
    }
    for (int i = 0; i < len; i++) {
      uint8_t n = (self->head + 1) % sizeof(self->rx);
      if (n != self->tail) { self->rx[self->head] = d[i]; self->head = n; }
    }
  }
  static void onSent(SEND_CB_ARG, esp_now_send_status_t s) {
    sent++;
    if (s != ESP_NOW_SEND_SUCCESS) failed++;
  }
  void enqueue(const uint8_t* p, size_t n) {
    if (!outQueue || !n) return;
    RadioMsg m; m.len = (uint8_t)n; memcpy(m.data, p, n);
    if (xQueueSend(outQueue, &m, 0) != pdTRUE) queueDropped++;
  }

 public:
  static volatile uint32_t sent, failed;
  volatile uint32_t queueDropped = 0, cmdDropped = 0;
  QueueHandle_t outQueue = nullptr, cmdQueue = nullptr;
  bool up = false;

  void begin(QueueHandle_t commands) {
    lock = xSemaphoreCreateMutex();
    outQueue = xQueueCreate(RADIO_QUEUE_LEN, sizeof(RadioMsg));
    cmdQueue = commands;
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    if (esp_now_init() != ESP_OK) { Serial.println("radio: ESP-NOW init FAILED"); return; }
    esp_now_peer_info_t p = {};
    memcpy(p.peer_addr, GROUND_MAC, 6);
    p.channel = 0;
    p.encrypt = false;
    esp_now_add_peer(&p);
    self = this;
    esp_now_register_recv_cb(onRecv);
    esp_now_register_send_cb(onSent);
    up = true;
    uint8_t mac[6] = {0};
    esp_wifi_get_mac(WIFI_IF_STA, mac);   // WiFi.macAddress() reads 00:00:.. here on core 3.x
    Serial.printf("radio: up, my MAC %02X:%02X:%02X:%02X:%02X:%02X (expect F8:B3:B7:30:33:40)\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  }

  // Print interface: USB immediately, radio one line per packet via the queue.
  size_t write(uint8_t c) override {
    Serial.write(c);
    tx[txn++] = c;
    if (c == '\n' || txn >= sizeof(tx)) { enqueue((const uint8_t*)tx, txn); txn = 0; }
    return 1;
  }
  void lockText() { if (lock) xSemaphoreTake(lock, portMAX_DELAY); }
  void unlockText() { if (lock) xSemaphoreGive(lock); }

  void sendBinary(const void* p, size_t n) { enqueue((const uint8_t*)p, n); }

  // radioTask body: the ONLY place esp_now_send is called.
  void serviceForever() {
    RadioMsg m;
    for (;;) {
      if (xQueueReceive(outQueue, &m, portMAX_DELAY) == pdTRUE && up) {
        esp_now_send(GROUND_MAC, m.data, m.len);
        vTaskDelay(pdMS_TO_TICKS(RADIO_SETTLE_MS));   // 15 ms settle, do not reduce
      }
    }
  }

  int available() {
    int n = (int)head - (int)tail;
    if (n < 0) n += sizeof(rx);
    return n + Serial.available();
  }
  int read() {
    if (head != tail) {
      char c = rx[tail];
      tail = (tail + 1) % sizeof(rx);
      return c;
    }
    return Serial.read();
  }
};

RadioLink* RadioLink::self = nullptr;
volatile uint32_t RadioLink::sent = 0;
volatile uint32_t RadioLink::failed = 0;
