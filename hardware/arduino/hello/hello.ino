// Kite: the Arduino says hello over USB serial, once a second.
// No processing yet: this only proves the board and the computer can talk.
//
// Upload with the Arduino IDE (Tools > Board, Tools > Port), then either open
// Tools > Serial Monitor at 115200 baud or run hardware/arduino/read_serial.py.
// Works on Uno, Nano, Mega, Leonardo, ESP32 and ESP8266 (anything with Serial).

const unsigned long BAUD = 115200;
const unsigned long EVERY_MS = 1000;

unsigned long count = 0;
unsigned long lastSend = 0;

void setup() {
  Serial.begin(BAUD);
  // Boards with native USB (Leonardo, Micro, some ESP32-S2/S3) wait for the port to open,
  // but never longer than 2 s so the board also runs with nothing connected.
  unsigned long start = millis();
  while (!Serial && millis() - start < 2000) {}
  Serial.println("kite: arduino ready");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSend >= EVERY_MS) {
    lastSend = now;
    count++;
    // One line per message, so the computer can read it line by line.
    Serial.print("hello ");
    Serial.println(count);
  }
}
