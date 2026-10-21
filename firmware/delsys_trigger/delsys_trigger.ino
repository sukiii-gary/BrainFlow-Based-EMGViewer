// Arduino Nano / ATmega328P, 16 MHz, 5 V.
// D2: combined START/STOP pulse; D3: held LOW as the BNC return.
// Serial: 115200 baud, newline-terminated ASCII (LF or CRLF).
// Reset clears the last event ID. HELLO does not clear it.
#include <Arduino.h>
#include <avr/interrupt.h>
#include <util/atomic.h>
#include <stdint.h>
#include <string.h>

#if !defined(__AVR_ATmega328P__) || F_CPU != 16000000UL
#error "This firmware requires an ATmega328P running at 16 MHz."
#endif

static const uint8_t TRIGGER_PIN = 2;
static const uint8_t RETURN_PIN = 3;
static const uint16_t PULSE_TICKS = 25000; // 16 MHz / 64 * 0.100 s
static const uint32_t MIN_TRIGGER_GAP_US = 2000000UL;
static const size_t LINE_CAPACITY = 48;
static char line[LINE_CAPACITY];
static size_t lineLength = 0;
static bool discardLine = false;
static bool invalidLine = false;
static volatile bool pulseActive = false;
static uint32_t lastId = 0;
static uint32_t lastRiseUs = 0;
static bool lastWasStart = false;

ISR(TIMER1_COMPA_vect) {
  PORTD &= ~_BV(PD2);
  TCCR1B = 0;
  TIMSK1 = 0;
  pulseActive = false;
}

static bool parseId(const char *text, uint32_t &id) {
  if (*text == '\0') return false;
  uint32_t value = 0;
  while (*text != '\0') {
    if (*text < '0' || *text > '9') return false;
    const uint8_t digit = *text++ - '0';
    if (value > (UINT32_MAX - digit) / 10UL) return false;
    value = value * 10UL + digit;
  }
  if (value == 0) return false;
  id = value;
  return true;
}

static void acknowledge(bool duplicate) {
  Serial.print(F("ACK "));
  Serial.print(lastId);
  Serial.print(lastWasStart ? F(" START ") : F(" STOP "));
  Serial.print(duplicate ? F("DUP ") : F("FIRED "));
  Serial.println(lastRiseUs);
}

static void handleLine(const char *command) {
  if (strcmp(command, "HELLO") == 0) {
    Serial.println(F("READY DELSYS_SYNC 2"));
    return;
  }
  if (strcmp(command, "STATUS") == 0) {
    const bool active = pulseActive;
    Serial.print(active ? F("STATUS PULSE ") : F("STATUS IDLE "));
    Serial.println(lastId);
    return;
  }

  bool isStart;
  const char *idText;
  if (strncmp(command, "START ", 6) == 0) {
    isStart = true;
    idText = command + 6;
  } else if (strncmp(command, "STOP ", 5) == 0) {
    isStart = false;
    idText = command + 5;
  } else {
    Serial.println(F("ERR COMMAND"));
    return;
  }
  uint32_t id;
  if (!parseId(idText, id)) {
    Serial.println(F("ERR ID"));
    return;
  }
  if (id == lastId && isStart == lastWasStart) {
    acknowledge(true);
    return;
  }
  if (id <= lastId) {
    Serial.println(F("ERR STALE"));
    return;
  }
  if (pulseActive) {
    Serial.println(F("ERR BUSY"));
    return;
  }
  if (lastId != 0 && (uint32_t)(micros() - lastRiseUs) < MIN_TRIGGER_GAP_US) {
    Serial.println(F("ERR TOO_SOON"));
    return;
  }

  lastId = id;
  lastWasStart = isStart;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    TCCR1A = 0;
    TCCR1B = 0;
    TCNT1 = 0;
    OCR1A = PULSE_TICKS - 1;
    TIFR1 = _BV(OCF1A);
    pulseActive = true;
    PORTD |= _BV(PD2);
    lastRiseUs = micros(); // Local software timestamp, not a host timestamp.
    TIMSK1 = _BV(OCIE1A);
    TCCR1B = _BV(WGM12) | _BV(CS11) | _BV(CS10);
  }
  // Pulse termination remains interrupt-driven even if Serial.print blocks.
  acknowledge(false);
}

void setup() {
  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(RETURN_PIN, LOW);
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(RETURN_PIN, OUTPUT);
  TCCR1A = 0;
  TCCR1B = 0;
  TIMSK1 = 0;
  Serial.begin(115200);
  Serial.println(F("READY DELSYS_SYNC 2"));
}

void loop() {
  while (Serial.available() > 0) {
    const char value = static_cast<char>(Serial.read());
    if (value == '\n') {
      if (discardLine) {
        Serial.println(F("ERR LINE_TOO_LONG"));
      } else if (invalidLine) {
        Serial.println(F("ERR COMMAND"));
      } else {
        if (lineLength > 0 && line[lineLength - 1] == '\r') --lineLength;
        line[lineLength] = '\0';
        handleLine(line);
      }
      lineLength = 0;
      discardLine = false;
      invalidLine = false;
    } else if (!discardLine) {
      if ((value < ' ' && value != '\r') || value > '~') invalidLine = true;
      if (lineLength == LINE_CAPACITY - 1) {
        discardLine = true;
      } else {
        line[lineLength++] = value;
      }
    }
  }
}
