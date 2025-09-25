import math
import cv2
import mediapipe as mp

def calculate_angle(a, b, c):
    # a, b, c sont des tuples (x, y)
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    if mag_ab * mag_cb == 0:
        return 0

    angle_rad = math.acos(dot_product / (mag_ab * mag_cb))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def get_xy(point):
    if isinstance(point, tuple) or isinstance(point, list):
        return point[0], point[1]
    else:
        return point.x, point.y

def distance_2d(a, b):
    ax, ay = get_xy(a)
    bx, by = get_xy(b)
    return math.sqrt((bx - ax)**2 + (by - ay)**2)

def calibration():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)  # ou chemin vers vidéo
    i = 0
    cal_left = 0
    cal_right = 0
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

                # Récupérer les points clés visibles du haut du corps
                try:
                    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                except IndexError:
                    # Certains points ne sont pas détectés, on ignore cette frame
                    continue

                cal_left = distance_2d(left_shoulder, left_hip)
                cal_right = distance_2d(right_shoulder, right_hip)
                cv2.putText(image, 'Prendre une posture correcte (Calibration en cours...)', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Affichage des résultats
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Calibration Posture', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"Calibration terminée: Gauche={cal_left:.2f}, Droite={cal_right:.2f}")
    return cal_left, cal_right