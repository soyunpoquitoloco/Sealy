# Sealy 🦭  
**Playful robotic companion for everyday productivity (posture • fatigue • phone distraction)**

Sealy is a small, seal-like desktop companion that supports focus through **low-demand, nonverbal mirroring**. Instead of notifications, scores, or “warnings,” Sealy uses **slow, peripheral, socially legible motions**:
- **Posture drift →** Sealy subtly *slumps* to mirror you  
- **Fatigue cues →** Sealy *gradually closes its eyelids*  
- **Phone distraction →** Sealy *turns away* from its “fishing hole” by rotating its base

<img width="648" height="197" alt="Capture d’écran 2026-01-28 à 01 43 39" src="https://github.com/user-attachments/assets/7b37b1f8-72b7-4982-8c88-93978849a1e2" />

This repo contains the **software**, **hardware notes**, and **fabrication assets** used to build and run Sealy.  
(Technical and study protocol details are documented in the accompanying paper.) :contentReference[oaicite:0]{index=0}

---

## Contents
- [Project at a glance](#project-at-a-glance)
- [Prototype iterations](#prototype-iterations)
- [Repository layout](#repository-layout)
- [System overview](#system-overview)
- [Bill of materials](#bill-of-materials)
- [Fabrication & assembly](#fabrication--assembly)
- [Software setup](#software-setup)
- [How to use](#how-to-use)
- [Preliminary lab study & evaluation plan](#preliminary-lab-study--evaluation-plan)
- [Media](#media)
- [Citation](#citation)

---

## Project at a glance
**Design goal:** Reduce brief self-regulation lapses (posture drift, fatigue, phone checks) with **ambient co-regulation** cues that feel **supportive, not supervisory**. :contentReference[oaicite:1]{index=1}

**Key design constraints:**
- perceivable in peripheral vision  
- no speech, alerts, or explicit “correction”  
- slow transitions, small amplitudes  
- avoid “being watched” impressions via on-device inference and minimal logging :contentReference[oaicite:2]{index=2}

---

## Prototype iterations

There are **four physical prototypes**. 

<img width="972" height="451" alt="Capture d’écran 2026-01-28 à 01 42 11" src="https://github.com/user-attachments/assets/81a566bf-e288-450e-b8f0-e796fcd7b626" />


### Prototype 1 — Origami scale mock
**What it was:** Paper origami seal.

**Why:**  
- validate **desk footprint**, approximate height, and “presence” in peripheral vision  
- test early **role framing** (“pet” vs “monitor”) without committing to materials or mechanics  
- quickly iterate silhouette (roundness, head size, flipper placement) for “baby-schema” readability :contentReference[oaicite:3]{index=3}

**What we learned → what changed next:**  
Origami is great for proportions but can’t validate **mount points**, servo routing, or deformation. Next step required a rigid, printable geometry to start integrating mechanisms.

---

### Prototype 2 — 3D printed PLA body
**What it was:** Full PLA print of Sealy.

**Why:**  
- confirm **3D model correctness** (surface continuity, eye/face legibility, printability)  
- test **mechanical integration**: internal clearances, routing for eyelids/posture line, base interface  
- verify “desk object” presence and camera placement constraints

**What we learned → what changed next:**  
PLA validates geometry, but the interaction goal includes a **soft companion feel** (tactility + deformation). Rigid plastic reads more like a gadget/toy shell and limits lifelike slumping. Next step: silicone casting.

---

### Prototype 3 — Silicone cast (full silicone, rigid feel)
**What it was:** Silicone part cast in a 3D printed mold, but **the body is fully silicone (solid)** → results in *low flexibility* for expressive deformation.

**Why:**  
- explore **tactile affordances** (softness, warmth, “pet-like” handling)  
- attempt deformation-driven motions (slump/breathing) via material compliance rather than only joints

**What we learned → what changed next:**  
A **fully-solid silicone** body tends to be:
- **too stiff / too massive** to deform visibly with small servos
- harder to create predictable motion (deformation becomes “mushy” or barely noticeable)
- difficult for cable/servo integration (channels, anchors, eyelid linkage)

So the next iteration needed **softness at the surface** *without* losing a controllable internal structure.

---

### Prototype 4 — Silicone “skin” (Rebound 25) over PLA core (flexible outer layer)
**What it was:** A PLA printed model used to create a silicone skin using **Rebound 25**, keeping **only the external silicone layer** (more flexible), with an internal rigid structure for mounting.

**Why this solved Prototype 3’s limits:**
- **Soft-touch exterior** preserves pet-like tactility and “cute” embodiment
- **Rigid inner core** provides:
  - stable servo mounts and anchors (posture line, eyelid linkage)
  - consistent kinematics (repeatable slump angle, eyelid closure range)
- **Lower actuation load**: servos deform a thin skin far more effectively than a solid silicone block

This aligns with Sealy’s technical intent: expressive, low-amplitude motion that remains readable and calm. :contentReference[oaicite:4]{index=4}

---

## Repository layout

```
Companion/
├── README.md                          # This file
├── yolov8n.pt                         # YOLOv8n model weights
│
├── Programme PC/                      # PC-side programs
│   ├── detect_phone.py                # Phone detection module
│   ├── Emotions.py                    # Emotion detection (legacy)
│   ├── Posture_old.py                 # Old posture detection
│   ├── yolov8n.pt                     # Model weights (copy)
│   │
│   ├── Digital/                       # Digital prototype
│   │   ├── Animation.py               # Animation logic
│   │   ├── Detection.py               # Detection pipeline
│   │   └── main.py                    # Entry point
│   │
│   ├── Programme_general/             # Main integrated system
│   │   ├── main.py                    # Main detection loop (posture + phone)
│   │   ├── posture.py                 # Posture calibration & detection
│   │   ├── websocket_client.py        # WebSocket communication to ESP32
│   │   └── yolov8n.pt                 # Model weights
│   │
│   └── tiredness_detection/           # Fatigue detection system
│       ├── main.py                    # Entry point
│       ├── config.py                  # Configuration parameters
│       ├── fatigue_detector.py        # Core fatigue detection
│       ├── metrics.py                 # EAR, MAR metrics
│       ├── requirements.txt           # Python dependencies
│       └── README.md                  # Module documentation
```

---

## System overview

### Sensing (PC / webcam)
Sealy uses on-device computer vision to estimate:
- **Posture** via **MediaPipe Pose** (baseline calibration, then compare shoulder/spine deviation)
- **Fatigue** via **MediaPipe Face Mesh** (EAR, MAR, PERCLOS-style eye closure)
- **Phone use** via pose + **YOLOv8n** object detection :contentReference[oaicite:6]{index=6}

### State → behavior mapping (robot)
- **Posture drift →** servo-driven “slump” (mirrors direction/weight)
  
<img width="1016" height="643" alt="Capture d’écran 2026-01-28 à 01 46 07" src="https://github.com/user-attachments/assets/d46f6c4a-a0c5-4195-b210-8be49647168e" />

- **Fatigue →** eyelids gradually close; tempo reduces
  
- **Phone distraction →** base rotates away from the fishing hole :contentReference[oaicite:7]{index=7}
  
<img width="967" height="538" alt="Capture d’écran 2026-01-28 à 01 45 13" src="https://github.com/user-attachments/assets/0b8a2634-a277-4255-96c0-761df5a00654" />


### Comms
PC runs a processing loop and transmits state to the device via a **WebSocket**-style connection for synchronized responses. :contentReference[oaicite:8]{index=8}

---

## Bill of materials

### Electronics
- Webcam (PC-facing; can be laptop webcam)
- **1× stepper motor** + driver (for rotating base)
- **2× servo motors**
  - posture/breathing (pull line)
  - eyelids (wire linkage)
- Controller: **ESP32 Devkit** 
- Power supply 5V
- Wires, connectors

<img width="526" height="256" alt="Capture d’écran 2026-01-28 à 01 49 39" src="https://github.com/user-attachments/assets/73438a94-4a17-4cc0-b087-dc9652bd67ce" />


### Mechanical
- Rotating platform + gear assembly (for ice-floe metaphor)
- Fishing-hole top reference (visual orientation cue)
- Trasnparent fishing line (posture actuation, eyelids)
  
<img width="1091" height="698" alt="Capture d’écran 2026-01-28 à 01 50 16" src="https://github.com/user-attachments/assets/373e9a77-0816-4840-a1e8-bf5f49d122fd" />

### Fabrication
- PLA filament (body core, base components, molds as needed)
- Silicone: **Rebound 25** (skin)
- Mold release spray + mixing supplies (cups, stir sticks, gloves)

---

## Fabrication & assembly

### A) Build the body (Prototype 4 recommended)
1. **3D print** the Sealy body core in PLA (and any internal brackets).
2. Prepare the **mold workflow** for a thin silicone skin:
   - use the PLA model as the master
   - apply mold release
3. Mix and brush/apply **Rebound 25** in layers to achieve a uniform skin thickness.
4. Demold the skin.

<img width="411" height="698" alt="Capture d’écran 2026-01-28 à 01 54 27" src="https://github.com/user-attachments/assets/d7ef77fe-bcd1-4406-aa74-ef242d6e9406" />


**Why this works:** thin silicone skin gives visible deformation under low torque, while the PLA core allowed to keep motion repeatable.

### B) Assemble the base (rotation)
1. Mount the stepper motor, and platform.
2. Ensure the base can rotate smoothly through the intended angle range (e.g., ~30° for “turn away”). :contentReference[oaicite:10]{index=10}
3. Align Sealy with the platform to make it face the fishing hole.

### C) Add actuators
- **Posture servo:** attach fishing line to a secure anchor on Sealy’s back; route line to servo horn so it can “pull” a controlled slump.
- **Eyelid servo:** connect eyelids via thin wires so the servo can close them gradually (aim ~50% closure for readability). :contentReference[oaicite:11]{index=11}

### D) Wire + power
- Keep stepper power isolated/adequate; servos can brown-out controllers if underpowered.
- Strain-relief all moving wires (rotation + servo motion).

---

## Software setup

### 1) PC-side (vision + state server)
**Expected dependencies:**
- Python 3.9+
- `opencv-python`
- `mediapipe`
- `ultralytics` (YOLOv8)
- `numpy`
- `websockets` (or equivalent)

**Installation:**
```bash
pip install opencv-python mediapipe ultralytics numpy websockets
```

**Directory options:**

- **`Programme_general/`** (recommended): integrated posture + phone detection with WebSocket communication
  ```bash
  cd "Programme PC/Programme_general"
  python main.py
  ```

- **`tiredness_detection/`**: standalone fatigue detection system
  ```bash
  cd "Programme PC/tiredness_detection"
  pip install -r requirements.txt
  python main.py
  ```
  
  Optional arguments:
  - `--camera-index N` — select camera device (default: 0)
  - `--no-calibration` — skip calibration, use default thresholds
  - `--ear-threshold 0.20` — customize eye aspect ratio threshold
  - `--log-level DEBUG` — set logging verbosity

**Configuration:**
- On first run, the system will **calibrate** your neutral posture (stay upright for ~5 seconds)
- Adjust detection thresholds in `config.py` (fatigue module) or directly in `main.py` scripts
- Ensure YOLOv8n weights (`yolov8n.pt`) are present in the working directory

---

### 2) Device-side (ESP32)

**Expected dependencies:**
- Arduino IDE or PlatformIO for ESP32 development
- ESP32 board support package
- Required libraries:
  - `ESP32Servo` (for servo control)
  - `AccelStepper` or `Stepper` (for stepper motor)
  - `WebSocketsServer` or `AsyncWebSocket` (for WebSocket communication)

**Setup:**
1. Flash firmware to ESP32:
   - Open the ESP32 firmware project in Arduino IDE or PlatformIO
   - Configure WiFi credentials and WebSocket port
   - Upload to ESP32 board

2. Configure PC connection:
   - Update WebSocket endpoint in `Programme_general/websocket_client.py` with your ESP32's IP address
   - Ensure both PC and ESP32 are on the same network

**Hardware wiring:**
- Connect servos to ESP32 GPIO pins (servo1 for slumping : 27, servo2 for eyelids : )
- Connect stepper motor driver to appropriate pins (Pin1 : 13, Pin2 : 26, Pin3 : 25, Pin4 : 33) 
- Use external power supply for motors 5V
---

## How to use

### First-time setup
1. **Build the physical prototype** (see [Fabrication & assembly](#fabrication--assembly))
2. **Install software** on both PC and ESP32 (see [Software setup](#software-setup))
3. **Connect hardware**: plug in webcam, power on Sealy, establish PC ↔ ESP32 communication
4. **Calibrate**: run the PC program and hold good posture during calibration countdown

### Daily use workflow
1. **Start the ESP32** (ensure it's powered on and connected to WiFi)
2. **Start PC detection**:
   ```bash
   cd "Programme PC/Programme_general"
   python main.py
   ```
3. **Position yourself** in front of the webcam, with Sealy visible in your peripheral vision
4. **Work naturally** — Sealy will respond to:
   - **Posture drift**: if you slump or lean, Sealy mirrors the motion

<img width="305" height="483" alt="Capture d’écran 2026-01-28 à 02 03 16" src="https://github.com/user-attachments/assets/ef377eea-e30e-4773-8d45-d44310097083" />


   - **Fatigue signs**: if your eyes close repeatedly, Sealy's eyelids droop

<img width="545" height="412" alt="Capture d’écran 2026-01-28 à 02 03 48" src="https://github.com/user-attachments/assets/f9ee4733-f6c5-416d-846a-ae956ce6440e" />


   - **Phone checks**: if you pick up your phone, Sealy rotates away from the fishing hole

<img width="416" height="475" alt="Capture d’écran 2026-01-28 à 02 04 12" src="https://github.com/user-attachments/assets/e0f07417-c675-4b09-93da-35c20a8fc567" />


### Interpreting Sealy's cues
- **Slump motion** → gentle reminder to adjust posture (not a "correction," just mirroring)
- **Eyelid closure** → fatigue detected; consider a break or eye rest
- **Base rotation** → phone distraction noticed; Sealy "looks away" to create ambient awareness

### Stopping / pausing
- Press `q` or `Ctrl+C` in the terminal to stop detection
- Device motors return to neutral positions
- No data is logged by default (privacy by design)

---

## Preliminary lab study & evaluation plan

### Study goals
Evaluate whether **nonverbal ambient cues** from Sealy can:
1. Reduce frequency/duration of posture lapses, fatigue incidents, and phone checks
2. Feel supportive rather than supervisory (subjective perception)
3. Maintain effectiveness over multi-day usage (habituation assessment)

### Study design (future work)
- **Participants**: 4 workers (students/office workers)
- **Protocol**: 
  - Baseline session (no Sealy): 15-20 min focused task
  - Sealy session: 15-20 min focused task with Sealy active
  - Counterbalanced order (half start with Sealy, half without)
- **Measures**:
  - **Quantitative**: posture deviation frequency, phone pickup count, eye closure duration (via logs)
  - **Qualitative**: post-session interview (NASA-TLX workload, perceived support vs. surveillance)


### Next steps
- Test on 4 participant
- Long-term deployment (1 week) to assess habituation
- Compare Sealy to explicit notifications (e.g., popup alerts) for perceived intrusiveness

---

## Media

### Photos

<img width="991" height="698" alt="Capture d’écran 2026-01-28 à 01 56 21" src="https://github.com/user-attachments/assets/de03eb11-d1a2-4d1c-893e-a65be9897c82" />

<img width="991" height="698" alt="Capture d’écran 2026-01-28 à 01 56 45" src="https://github.com/user-attachments/assets/f5248241-8acc-4f07-b57d-e34adf70d022" />


### Demo videos

https://github.com/user-attachments/assets/3ee4d5ab-db28-43dc-951a-25afe59001c0



### Design process

<img width="634" height="394" alt="Capture_d_écran_2026-01-22_à_23 58 56-removebg-preview" src="https://github.com/user-attachments/assets/caf86437-e3d8-4fac-8a37-278a2cfef1d0" />

<img width="634" height="394" alt="Capture_d_écran_2026-01-22_à_23 59 40-removebg-preview" src="https://github.com/user-attachments/assets/90a639d6-e530-4ade-a381-e57c18d820f7" />


