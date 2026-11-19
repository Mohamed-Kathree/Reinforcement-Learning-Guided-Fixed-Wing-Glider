/*
 * hw_servo.h -- four servo outputs on the MCPWM peripheral (group 0).
 *
 * WHY NOT ESP32Servo/LEDC (measured 19 Nov 2026, on battery): each
 * writeMicroseconds() -> ledcWrite() took ~10 ms and stalled core 1 with the
 * IMU interrupt blocked; with four servos updating at 50 Hz the 228 Hz control
 * loop fell to 50-130 Hz with gaps of up to 43 ms. The bench sketch still uses
 * ESP32Servo (verified there, where timing does not matter).
 *
 * MCPWM: one 1 MHz timer, 20 000-tick period (50 Hz); two operators, each with
 * two comparator/generator pairs. Output goes HIGH at timer zero and LOW at the
 * compare value, so compare ticks = pulse width in microseconds. New compare
 * values are written to a shadow register and latched by hardware at the next
 * timer zero (update_cmp_on_tez): a write never waits and never produces a
 * runt pulse. Writes are clamped to the absolute envelope RLG_ABS_MIN/MAX_US.
 */
#pragma once
#include "driver/mcpwm_prelude.h"
#include "src/protocol/rlg_protocol.h"

static mcpwm_timer_handle_t srvTimer = nullptr;
static mcpwm_oper_handle_t srvOper[2] = {nullptr, nullptr};
static mcpwm_cmpr_handle_t srvCmpr[RLG_NUM_SERVOS] = {nullptr};
static mcpwm_gen_handle_t srvGen[RLG_NUM_SERVOS] = {nullptr};
static bool srvOk = false;

static uint16_t servoClamp(uint16_t us) {
  return us < RLG_ABS_MIN_US ? RLG_ABS_MIN_US : (us > RLG_ABS_MAX_US ? RLG_ABS_MAX_US : us);
}

// Returns false (and logs which step) if the driver refused anything.
static bool servoBegin(const int pins[RLG_NUM_SERVOS], const uint16_t initUs[RLG_NUM_SERVOS], Print* log) {
  mcpwm_timer_config_t tc = {};
  tc.group_id = 0;
  tc.clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT;
  tc.resolution_hz = 1000000;          // 1 tick = 1 us
  tc.period_ticks = 20000;             // 50 Hz
  tc.count_mode = MCPWM_TIMER_COUNT_MODE_UP;
  if (mcpwm_new_timer(&tc, &srvTimer) != ESP_OK) { log->println("servo: timer FAILED"); return false; }
  for (int o = 0; o < 2; o++) {
    mcpwm_operator_config_t oc = {};
    oc.group_id = 0;
    if (mcpwm_new_operator(&oc, &srvOper[o]) != ESP_OK ||
        mcpwm_operator_connect_timer(srvOper[o], srvTimer) != ESP_OK) {
      log->println("servo: operator FAILED"); return false;
    }
  }
  for (int i = 0; i < RLG_NUM_SERVOS; i++) {
    mcpwm_oper_handle_t op = srvOper[i / 2];
    mcpwm_comparator_config_t cc = {};
    cc.flags.update_cmp_on_tez = true;
    mcpwm_generator_config_t gc = {};
    gc.gen_gpio_num = pins[i];
    if (mcpwm_new_comparator(op, &cc, &srvCmpr[i]) != ESP_OK ||
        mcpwm_new_generator(op, &gc, &srvGen[i]) != ESP_OK ||
        mcpwm_comparator_set_compare_value(srvCmpr[i], servoClamp(initUs[i])) != ESP_OK ||
        mcpwm_generator_set_action_on_timer_event(srvGen[i],
            MCPWM_GEN_TIMER_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, MCPWM_TIMER_EVENT_EMPTY, MCPWM_GEN_ACTION_HIGH)) != ESP_OK ||
        mcpwm_generator_set_action_on_compare_event(srvGen[i],
            MCPWM_GEN_COMPARE_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, srvCmpr[i], MCPWM_GEN_ACTION_LOW)) != ESP_OK) {
      log->printf("servo: channel %d (GPIO %d) FAILED\n", i, pins[i]); return false;
    }
  }
  if (mcpwm_timer_enable(srvTimer) != ESP_OK ||
      mcpwm_timer_start_stop(srvTimer, MCPWM_TIMER_START_NO_STOP) != ESP_OK) {
    log->println("servo: start FAILED"); return false;
  }
  srvOk = true;
  log->println("servo: MCPWM 50 Hz on GPIO 13/14/25/27");
  return true;
}

// Non-blocking: shadow register, latched by hardware at the next period.
static inline void servoWriteUs(int i, uint16_t us) {
  if (srvOk) mcpwm_comparator_set_compare_value(srvCmpr[i], servoClamp(us));
}
