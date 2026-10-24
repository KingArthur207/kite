// Kite: the Arduino UNO Q says hello to the computer, once a second.
// No processing yet: this only proves the board and the computer can talk.
//
// On the UNO Q, Serial is the microcontroller's pin-0/1 UART and never reaches the USB-C port.
// Monitor (from the Arduino_RouterBridge library) goes microcontroller -> the board's Linux side -> USB-C,
// and shows up in the Arduino IDE's Serial Monitor, App Lab's console, or `arduino-cli monitor`.
// Board: Tools > Board > Arduino UNO Q Board > Arduino UNO Q (FQBN arduino:zephyr:unoq).

#include <Arduino_RouterBridge.h>

const unsigned long EVERY_MS = 1000;

unsigned long count = 0;
unsigned long lastSend = 0;

void setup() {
  Bridge.begin();
  Monitor.begin();              // the baud rate has no effect over the bridge
  pinMode(LED_BUILTIN, OUTPUT);
  Monitor.println("kite: arduino uno q ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend >= EVERY_MS) {
    lastSend = now;
    count++;
    Monitor.print("hello ");
    Monitor.println(count);
    digitalWrite(LED_BUILTIN, count % 2);   // blink once per message, so you can see it's alive
  }
}
