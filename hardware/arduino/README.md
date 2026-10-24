# Arduino: hello over USB

The first step for an Arduino in Kite: the board sends `hello 1`, `hello 2`, ... once a second over the
USB cable, and the computer prints it. No processing yet.

| Board | Sketch | Why |
|---|---|---|
| **Arduino UNO Q** (ours) | [`hello_unoq/hello_unoq.ino`](hello_unoq/hello_unoq.ino) | On the UNO Q, `Serial` is the microcontroller's pin 0/1 UART and never reaches USB-C. `Monitor` from the `Arduino_RouterBridge` library goes through the board's Linux side to USB-C. |
| Uno R3/R4, Nano, Mega, Leonardo, ESP32, ESP8266 | [`hello/hello.ino`](hello/hello.ino) | Plain `Serial` at 115200 baud. |
| **UNO Q on a battery, over Wi-Fi** | [`unoq_wifi/`](unoq_wifi/README.md) | No cable or IDE after a one-time setup: its Linux side sends hello through the internet (works on eduroam). |

Both compile with `arduino-cli` (UNO Q: `arduino:zephyr:unoq`; Uno/Nano/Leonardo: `arduino:avr:*`).

## UNO Q on the Windows computer

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) (2.3.6 or newer).
2. **Tools > Board > Boards Manager**: install **Arduino UNO Q Boards** (package `arduino:zephyr`).
   **Sketch > Include Library > Manage Libraries**: install **Arduino_RouterBridge**.
3. Plug the UNO Q in with USB-C. Pick **Tools > Board > Arduino UNO Q Board > Arduino UNO Q** and its port under
   **Tools > Port** (a `COM` number that appears when you plug it in). Open `hello_unoq/hello_unoq.ino`, press **Upload**.
   The built-in LED blinks once per message.
4. See the messages, either way:
   - **Tools > Serial Monitor** (the baud setting doesn't matter for the UNO Q), or
   - in a terminal: `pip install pyserial`, then `python hardware\arduino\read_serial.py`

```
connected to COM5 at 115200 baud
14:02:11  kite: arduino uno q ready
14:02:12  hello 1
14:02:13  hello 2
```

`python hardware\arduino\read_serial.py --list` shows every serial port and marks the one that looks like an
Arduino. Only one program can hold the port at a time, so close the Serial Monitor before running the script
(and the other way round). If nothing appears in the Serial Monitor, check that the board is set to
**Arduino UNO Q**, not a generic Zephyr board.

## Other boards

Same steps with `hello/hello.ino` and your board selected. Clone Nanos (CH340 chip) and most ESP32 boards
(CP210x or CH340) need the USB driver for that chip on Windows; ESP32 also needs "esp32 by Espressif" in
Boards Manager.
