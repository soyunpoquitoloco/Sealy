import cv2
import mediapipe as mp
import math
import time
import sys
import platform
import numpy as np
from collections import deque
from ultralytics import YOLO
from websocket_client import send_message
# -----------------------------
# FONCTIONS UTILITAIRES DU PREMIER PROGRAMME
# -----------------------------
def get_xy(point):
    if isinstance(point, tuple) or isinstance(point, list):
        return point[0], point[1]
    else:
        return point.x, point.y

def distance_2d(a, b):
    ax, ay = get_xy(a)
    bx, by = get_xy(b)
    return math.sqrt((bx - ax)**2 + (by - ay)**2)

# -----------------------------
# CALIBRATION DE LA POSTURE
# -----------------------------
def calibration():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    i = 0
    cal_chin = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and i < 200:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture de la caméra ou vidéo")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                except IndexError:
                    continue

                cal_chin = distance_2d(nose, right_shoulder)
                cv2.putText(image, 'Prendre une posture correcte', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Calibration Posture', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()
    return cal_chin

# -----------------------------
# OUVERTURE ROBUSTE DE LA WEBCAM (DU SECOND PROGRAMME)
# -----------------------------
def open_default_camera(preferred_width=1280, preferred_height=720, warmup_frames=5):
    system = platform.system().lower()
    if system.startswith('win'):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system.startswith('darwin'):  # macOS
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:  # Linux et autres
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        for cam_idx in range(0, 6):
            cap = cv2.VideoCapture(cam_idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)

            ok = False
            for _ in range(warmup_frames):
                time.sleep(0.03)
                ok, _frame = cap.read()
                if not ok:
                    break

            if ok:
                print(f"[OK] Webcam ouverte (index {cam_idx}, backend {backend}).")
                return cap

            cap.release()

    raise RuntimeError("Impossible d’ouvrir la webcam. Vérifie les permissions et qu’aucune autre appli ne l’utilise.")

# -----------------------------
# CHARGEMENT DES MODÈLES (DU SECOND PROGRAMME, PLUS POSE)
# -----------------------------
yolo = YOLO('yolov8n.pt')  # contient la classe "cell phone"
PHONE_CLASS = 67           # index COCO "cell phone"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose  # Ajouté pour la posture
mp_drawing = mp.solutions.drawing_utils  # Pour dessiner les landmarks

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Pour la posture

# -----------------------------
# OUTILS GÉOMÉTRIQUES (DU SECOND PROGRAMME)
# -----------------------------
def center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 180.0
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    c = np.clip(c, -1, 1)
    return math.degrees(math.acos(c))

# -----------------------------
# FILTRE TEMPOREL (HYSTÉRÉSIS) (DU SECOND PROGRAMME)
# -----------------------------
class TemporalGate:
    def __init__(self, up=0.7, down=0.4, up_time=1.0, down_time=0.6, fps=30):
        """
        up/down : seuils d'activation/désactivation sur la moyenne glissante
        up_time/down_time : durées (s) pendant lesquelles le score doit rester au-dessus/en dessous du seuil
        """
        self.up, self.down = up, down
        self.up_t, self.down_t = int(up_time*fps), int(down_time*fps)
        self.state = False
        self.buf = deque(maxlen=max(self.up_t, self.down_t))

    def update(self, s):
        self.buf.append(s)
        if not self.state:
            # Activation si moyenne assez haute pendant up_time
            if len(self.buf) == self.buf.maxlen and np.mean(self.buf) >= self.up:
                self.state = True
                self.buf.clear()
        else:
            # Désactivation si moyenne assez basse pendant down_time
            if len(self.buf) == self.buf.maxlen and np.mean(self.buf) <= self.down:
                self.state = False
                self.buf.clear()
        return self.state

gate = TemporalGate(up=0.7, down=0.4, up_time=1.0, down_time=0.6, fps=30)

# -----------------------------
# HISTORIQUE DU POUCE (DU SECOND PROGRAMME)
# -----------------------------
thumb_history = deque(maxlen=24)  # ~0.8 s à 30FPS

# -----------------------------
# FONCTION PRINCIPALE FUSIONNÉE
# -----------------------------
def main(cal_chin):
    # Ouvrir la caméra
    cap = open_default_camera(1280, 720)

    # Compteurs pour les comportements (supposant ~30 FPS, 10s = 300 frames)
    FPS_ESTIMATED = 10
    THRESHOLD_FRAMES = 10 * FPS_ESTIMATED  # 300 frames pour 10 secondes
    count_posture = 0
    count_distraction = 0
    printing_posture = False
    printing_distraction = False
    message = ""
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture de la caméra ou vidéo")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # 1) Détection de la posture (du premier programme)
            posture_bad = False
            results_pose = pose.process(rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                try:
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                except IndexError:
                    pass
                else:
                    length_chin = distance_2d(nose, right_shoulder)
                    chin_change = (length_chin - cal_chin) / cal_chin * 100
                    posture_bad = chin_change < -10

                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 2) Téléphone (YOLO)
            y = yolo.predict(source=rgb, imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
            phone_boxes = []
            for box, cls, conf in zip(y.boxes.xyxy.cpu().numpy(),
                                      y.boxes.cls.cpu().numpy(),
                                      y.boxes.conf.cpu().numpy()):
                if int(cls) == PHONE_CLASS:
                    x1, y1, x2, y2 = box
                    phone_boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(frame, f'phone {conf:.2f}', (int(x1), int(y1)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 3) Mains (MediaPipe)
            hand_landmarks = []
            res_h = hands.process(rgb)
            if res_h.multi_hand_landmarks:
                for hl in res_h.multi_hand_landmarks:
                    pts = [(lm.x*w, lm.y*h) for lm in hl.landmark]  # 21 pts
                    hand_landmarks.append(pts)
                    for p in pts:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (255,200,0), -1)

            # 4) Visage + orientation
            face_center = None
            face_forward = None
            res_f = face.process(rgb)
            if res_f.multi_face_landmarks:
                fl = res_f.multi_face_landmarks[0]
                pts = [(lm.x*w, lm.y*h) for lm in fl.landmark]
                left_eye = pts[33]; right_eye = pts[263]; nose_tip = pts[1]; chin = pts[152]
                face_center = ((left_eye[0]+right_eye[0]+nose_tip[0]+chin[0])/4,
                               (left_eye[1]+right_eye[1]+nose_tip[1]+chin[1])/4)
                face_forward = (nose_tip[0] - (left_eye[0]+right_eye[0])/2,
                                nose_tip[1] - (left_eye[1]+right_eye[1])/2)
                cv2.circle(frame, (int(face_center[0]), int(face_center[1])), 3, (0,0,255), -1)

            # 5) Features pour le téléphone
            phone_present = 1.0 if len(phone_boxes) > 0 else 0.0
            min_hand_dist = 1e9
            phone_center = None
            if phone_boxes:
                phone_center = center(phone_boxes[0])  # on prend la 1ère boîte
            if phone_center and hand_landmarks:
                for hl in hand_landmarks:
                    for idx in [0, 5, 9, 13, 17, 4]:
                        d = l2(phone_center, hl[idx])
                        if d < min_hand_dist:
                            min_hand_dist = d
            if min_hand_dist == 1e9:
                min_hand_dist = 9999.0

            head_to_phone_angle = 180.0
            if face_center and phone_center and face_forward:
                v_to_phone = (phone_center[0]-face_center[0], phone_center[1]-face_center[1])
                head_to_phone_angle = angle_between(face_forward, v_to_phone)

            # activité du pouce
            thumb_speed = 0.0
            if hand_landmarks:
                thumbs = [hl[4] for hl in hand_landmarks]  # landmark du pouce
                thumb_history.append(thumbs[0] if thumbs else (None,None))
                if len(thumb_history) >= 2 and thumb_history[-2][0] is not None and thumb_history[-1][0] is not None:
                    dx = thumb_history[-1][0]-thumb_history[-2][0]
                    dy = thumb_history[-1][1]-thumb_history[-2][1]
                    thumb_speed = math.hypot(dx, dy) / max(w, h)
            thumb_activity = np.clip(thumb_speed*20.0, 0, 1)

            # 6) Score pour distraction (avec court-circuit "pas de téléphone" => 0.0)
            eps = 1e-3
            a1, a2, a3, a4 = 2.0, 1.2, 1.0, 0.7
            BIAS = -2.0  # pour éviter que la sigmoïde ne reste ~0.5 au repos

            def sigmoid(x):
                return 1/(1+math.exp(-x))

            if len(phone_boxes) == 0:
                # Aucun téléphone détecté -> force le score à 0.0 pour pousser l'état vers PAS
                use_score = 0.0
            else:
                inv_dist = 0.0
                if min_hand_dist != 9999.0:
                    inv_dist = 1.0/((min_hand_dist/(0.25*min(w, h))) + eps)
                gaze_ok = 1.0 if head_to_phone_angle < 25.0 else 0.0
                phone_present = 1.0  # on sait qu'il est présent
                raw = a1*inv_dist + a2*gaze_ok + a3*thumb_activity + a4*phone_present + BIAS
                use_score = sigmoid(raw)

            distraction = gate.update(use_score)

            # 7) Logique des compteurs et prints
            # Posture
            if posture_bad:
                count_posture += 1
                if count_posture >= THRESHOLD_FRAMES and not printing_posture:
                    printing_posture = True
                if printing_posture:
                    print("mauvaise posture")
                    message = message + "posture"
                else:
                    message = message + "not_posture"
            else:
                count_posture = 0
                printing_posture = False
                message = message + "not_posture"

            # Distraction
            if distraction:
                count_distraction += 1
                print(count_distraction)
                if count_distraction >= THRESHOLD_FRAMES and not printing_distraction:
                    printing_distraction = True
                if printing_distraction:
                    print("distrait")
                    message = message + " distracted"
                else:
                    message = message + " not_distracted"
            else:
                count_distraction = 0
                printing_distraction = False
                message = message + " not_distracted"
            send_message(message)
            message = ""
            # 8) Affichage sur la frame
            if posture_bad:
                cv2.putText(frame, 'Posture Avachie Detectee', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, 'Posture Assise Correcte', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            ui = f"score={use_score:.2f}  gaze<{head_to_phone_angle:.0f}deg  distN={0.0 if min_hand_dist==9999.0 else 1.0/((min_hand_dist/(0.25*min(w,h))) + 1e-3):.2f}  thumb={thumb_activity:.2f}  STATE={'SUR_TEL' if distraction else 'PAS'}"
            cv2.putText(frame, ui, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,255), 2)
            if phone_center:
                cv2.circle(frame, (int(phone_center[0]), int(phone_center[1])), 4, (0,255,255), -1)

            cv2.imshow('Detection Posture et Distraction', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Fermer proprement les solutions MediaPipe
        try:
            pose.close()
            hands.close()
            face.close()
        except Exception:
            pass

if __name__ == "__main__":
    cal_chin = calibration()
    main(cal_chin)