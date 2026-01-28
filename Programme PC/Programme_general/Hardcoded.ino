#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebSocketsServer.h>

// Déclaration du servomoteur
Servo servo1;

const char* ssid = "";
const char* password = "";

WebSocketsServer webSocket = WebSocketsServer(80);
String message;

void onWebSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED: {
            Serial.printf("Client %u connecté\n", num);
            webSocket.sendTXT(num, "w00t");
            break;
        }
        case WStype_DISCONNECTED: {
            Serial.printf("Client %u déconnecté\n", num);
            break;
        }
        case WStype_TEXT: {
            message = (char *)payload;
            Serial.println(message);
            break;
        }
        default: {
            break;
        }
    }
}

// Broches du moteur pas-à-pas
#define STEPPER_PIN_1 13
#define STEPPER_PIN_2 26
#define STEPPER_PIN_3 25
#define STEPPER_PIN_4 33

int step_number = 0;

void setup() {
    // Initialisation du servomoteur
    servo1.attach(27);
    delay(100);

    // Initialisation des broches du moteur pas-à-pas
    pinMode(STEPPER_PIN_1, OUTPUT);
    pinMode(STEPPER_PIN_2, OUTPUT);
    pinMode(STEPPER_PIN_3, OUTPUT);
    pinMode(STEPPER_PIN_4, OUTPUT);

    // Initialisation de la communication série
    Serial.begin(9600);
    Serial.println("Prêt !");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    Serial.println(WiFi.localIP());
    Serial.println("Starting WebSocket server...");
    webSocket.begin();
    webSocket.onEvent(onWebSocketEvent);
    Serial.println("Server ready");
}

bool keep_posture = false;
bool isRotating = false;
bool keep_distracted = false;

void loop() {
    webSocket.loop();

    // Vérifie les messages WebSocket (pas besoin de Serial.available())
    if ((message == "posture not_distracted" && keep_posture == true) ||(message == "posture distracted" && keep_posture == true)) {
        Serial.println("Activation du servomoteur...");
        // Tourne progressivement de 180° à 90°
        for (int angle = 180; angle >= 90; angle -= 1) {
            servo1.write(angle);
            delay(15);
        }
        //servo1.write(90);
        keep_posture = false;
    } 
    else if (message == "not_posture not_distracted" || message == "not_posture distracted") {
          servo1.write(180);
          keep_posture = true;
    }
    if ((message == "posture distracted" || message == "not_posture distracted") && (!isRotating && keep_distracted == false)) {
        Serial.println("Activation du moteur pas-à-pas (90° aller-retour)...");
        isRotating = true;
        rotate90();
        isRotating = false;
        keep_distracted = true;
    }
    else if ((message == "posture not_distracted" || message == "not_posture not_distracted") && (!isRotating && keep_distracted == true)) {
      //Serial.println("Activation du moteur pas-à-pas (90° aller-retour)...");
        isRotating = true;
        rotateBack();
        isRotating = false;
        keep_distracted = false;
    }
}

void rotate90() {
    const int steps_per_revolution = 200*8;
    const int steps_90 = steps_per_revolution / 2;

    for (int i = 0; i < steps_90; i++) {
        OneStep(false);
        delay(2);
    }
}

void rotateBack() {
  const int steps_per_revolution = 200*8;
  const int steps_90 = steps_per_revolution / 2;

  for (int i = 0; i < steps_90; i++) {
        OneStep(true);
        delay(2);
    }
}
void OneStep(bool dir) {
    if (dir) {
        switch (step_number) {
            case 0:
                digitalWrite(STEPPER_PIN_1, HIGH);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
            case 1:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, HIGH);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
            case 2:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, HIGH);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
            case 3:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, HIGH);
                break;
        }
    } else {
        switch (step_number) {
            case 0:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, HIGH);
                break;
            case 1:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, HIGH);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
            case 2:
                digitalWrite(STEPPER_PIN_1, LOW);
                digitalWrite(STEPPER_PIN_2, HIGH);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
            case 3:
                digitalWrite(STEPPER_PIN_1, HIGH);
                digitalWrite(STEPPER_PIN_2, LOW);
                digitalWrite(STEPPER_PIN_3, LOW);
                digitalWrite(STEPPER_PIN_4, LOW);
                break;
        }
    }
    step_number++;
    if (step_number > 3) {
        step_number = 0;
    }
}