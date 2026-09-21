// NeuroHand: ESP32 tendon-hand controller for 8x N20 via 4x DRV8833.
//
// The firmware owns the grasp primitives. The BCI only ever selects one of them
// by name, because per-finger intent is not recoverable from scalp EEG and the
// host should not be trusted with motor-level timing over a serial link.
//
// Protocol (newline-terminated ASCII, 115200):
//   G <primitive>   execute primitive        -> OK G <name> | ERR unknown primitive
//   R               release / open           -> OK R
//   S               hard stop (coast all)    -> OK S
//   P               heartbeat                -> OK P
//   ?               status                   -> ST <state> <primitive> <ms_remaining>
//   L <0..255>      global duty ceiling      -> OK L <n>
//
// Safety model, in order of importance:
//   1. Watchdog. Tendons are pull-only and the base N20 build has no position
//      feedback, so a motor left energised stalls against its tendon and burns.
//      If no line arrives for WATCHDOG_MS, every channel coasts and the hand is
//      reported open. The host must keep sending P.
//   2. Bounded actuation. Every move has a per-motor deadline from config.h; a
//      motor physically cannot be driven longer than its close_ms/open_ms.
//   3. Duty ramp. Duty rises over RAMP_MS to limit inrush on the shared buck,
//      since 8 stalling N20s can pull well over 10 A together.
//
// config.h is generated from config/hand.json by tools/gen_hand_config.py.

#include <Arduino.h>
#include "config.h"

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_ATTACH(pin, ch) ledcAttachChannel((pin), PWM_FREQ_HZ, PWM_BITS, (ch))
#else
  #define PWM_ATTACH(pin, ch) do { ledcSetup((ch), PWM_FREQ_HZ, PWM_BITS); \
                                   ledcAttachPin((pin), (ch)); } while (0)
#endif

static const uint16_t DUTY_MAX = (1u << PWM_BITS) - 1u;

struct MotorState {
  int8_t   dir;          // -1 release, 0 idle, +1 pull
  uint32_t start_ms;
  uint32_t until_ms;
};

static MotorState st[N_MOTORS];
static uint16_t duty_ceiling = DUTY_LIMIT;
static uint32_t last_rx_ms = 0;
static bool watchdog_tripped = false;
static const char* cur_prim = "open";
static String line;

static inline uint8_t chan_a(int i) { return (uint8_t)(2 * i); }
static inline uint8_t chan_b(int i) { return (uint8_t)(2 * i + 1); }

static void writeMotor(int i, int8_t dir, uint16_t duty) {
  bool inv = MOTORS[i].invert;
  int8_t d = inv ? (int8_t)-dir : dir;
  uint16_t a = 0, b = 0;
  if (d > 0)      a = duty;
  else if (d < 0) b = duty;
  ledcWrite(chan_a(i), a);
  ledcWrite(chan_b(i), b);
}

static void coastAll() {
  for (int i = 0; i < N_MOTORS; i++) {
    st[i].dir = 0;
    st[i].until_ms = 0;
    writeMotor(i, 0, 0);
  }
}

static int findPrim(const String& name) {
  for (int p = 0; p < N_PRIMS; p++)
    if (name == PRIMS[p].name) return p;
  return -1;
}

static void startPrim(int p) {
  uint32_t now = millis();
  for (int i = 0; i < N_MOTORS; i++) {
    int8_t d = PRIMS[p].dir[i];
    st[i].dir = d;
    st[i].start_ms = now;
    if (d == 0) {
      st[i].until_ms = 0;
      writeMotor(i, 0, 0);
    } else {
      // Release is given the longer window: without position feedback, full
      // opening depends on the passive return finishing its travel.
      uint16_t ms = (d > 0) ? MOTORS[i].close_ms : MOTORS[i].open_ms;
      st[i].until_ms = now + ms;
    }
  }
  cur_prim = PRIMS[p].name;
}

static uint32_t msRemaining() {
  uint32_t now = millis(), rem = 0;
  for (int i = 0; i < N_MOTORS; i++)
    if (st[i].dir != 0 && st[i].until_ms > now)
      rem = max(rem, st[i].until_ms - now);
  return rem;
}

static bool anyMoving() {
  for (int i = 0; i < N_MOTORS; i++)
    if (st[i].dir != 0) return true;
  return false;
}

static const char* stateName() {
  if (watchdog_tripped) return "coast";
  return anyMoving() ? "moving" : "idle";
}

static void handle(const String& raw) {
  String s = raw;
  s.trim();
  if (!s.length()) return;

  last_rx_ms = millis();
  if (watchdog_tripped) {
    // Require an explicit command after a trip; heartbeats alone must not re-arm.
    if (s == "P") { Serial.println("OK P"); return; }
    watchdog_tripped = false;
    Serial.println("EV watchdog_cleared");
  }

  char verb = s[0];
  String arg = (s.length() > 2) ? s.substring(2) : "";
  arg.trim();

  switch (verb) {
    case 'G': {
      int p = findPrim(arg);
      if (p < 0) { Serial.println("ERR unknown primitive " + arg); return; }
      startPrim(p);
      Serial.println("OK G " + arg);
      break;
    }
    case 'R': {
      int p = findPrim("open");
      if (p >= 0) startPrim(p); else coastAll();
      cur_prim = "open";
      Serial.println("OK R");
      break;
    }
    case 'S':
      coastAll();
      cur_prim = "open";
      Serial.println("OK S");
      break;
    case 'P':
      Serial.println("OK P");
      break;
    case '?':
      Serial.printf("ST %s %s %lu\n", stateName(), cur_prim, (unsigned long)msRemaining());
      break;
    case 'L': {
      long v = arg.toInt();
      if (v < 0) v = 0;
      if (v > DUTY_MAX) v = DUTY_MAX;
      duty_ceiling = (uint16_t)v;
      Serial.println("OK L " + String(v));
      break;
    }
    default:
      Serial.println("ERR bad verb " + String(verb));
  }
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N_MOTORS; i++) {
    PWM_ATTACH(MOTORS[i].in1, chan_a(i));
    PWM_ATTACH(MOTORS[i].in2, chan_b(i));
  }
  coastAll();
  last_rx_ms = millis();
  // Boot coasted, never energised: a brownout reset mid-grip must not resume pulling.
  Serial.printf("EV ready motors=%d prims=%d watchdog_ms=%d\n",
                N_MOTORS, N_PRIMS, WATCHDOG_MS);
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length()) { handle(line); line = ""; }
    } else if (line.length() < 64) {
      line += c;
    }
  }

  uint32_t now = millis();

  if (!watchdog_tripped && (now - last_rx_ms) > WATCHDOG_MS) {
    watchdog_tripped = true;
    coastAll();
    cur_prim = "open";
    Serial.println("EV watchdog_release");
  }

  for (int i = 0; i < N_MOTORS; i++) {
    if (st[i].dir == 0) continue;
    if (now >= st[i].until_ms) {
      st[i].dir = 0;
      writeMotor(i, 0, 0);
      continue;
    }
    uint32_t el = now - st[i].start_ms;
    uint32_t duty = duty_ceiling;
    if (RAMP_MS > 0 && el < (uint32_t)RAMP_MS) duty = duty * el / RAMP_MS;
    writeMotor(i, st[i].dir, (uint16_t)duty);
  }
}
