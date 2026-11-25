void setup() {
  Serial.begin(9600);
  pinMode(8, OUTPUT);  // Arduino pin for buzzer
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') digitalWrite(8, HIGH);
    if (c == '0') digitalWrite(8, LOW);
  }
}
