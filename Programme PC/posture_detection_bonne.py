import cv2
import mediapipe as mp
import math

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


def main(cal_left, cal_right, cal_chin):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)  # ou chemin vers vidéo

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
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
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                    left_ear = (landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y)
                    right_ear = (landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y)
                    left_eye = (landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y)
                    right_eye = (landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y)
                    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                except IndexError:
                    # Certains points ne sont pas détectés, on ignore cette frame
                    continue

                # Calculer l'angle entre la tête (nez), l'épaule gauche et l'épaule droite
                # Cela donne une idée de l'inclinaison de la tête par rapport aux épaules
                angle_tete = calculate_angle(left_shoulder, nose, right_shoulder)

                # Calculer l'angle entre les épaules et les hanches pour détecter un dos courbé
                angle_dos_gauche = calculate_angle(left_hip, left_shoulder, nose)
                angle_dos_droite = calculate_angle(right_hip, right_shoulder, nose)

                # Critères simples pour posture avachie assise
                posture_avachie = False
                length_left_side = distance_2d(left_shoulder, left_hip)
                length_right_side = distance_2d(right_shoulder, right_hip)
                length_chin = distance_2d(nose, right_shoulder)

                if length_chin < cal_chin -0.05:
                    posture_avachie = True

                # Affichage des résultats
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if posture_avachie:
                    cv2.putText(image, 'Posture Avachie Detectee', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(image, 'Posture Assise Correcte', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('Detection Posture Assise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def calibration(cal_left, cal_right, cal_chin):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)  # ou chemin vers vidéo
    i = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and i<200:
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
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                    left_ear = (landmarks[mp_pose.PoseLandmark.LEFT_EAR].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR].y)
                    right_ear = (landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y)
                    left_eye = (landmarks[mp_pose.PoseLandmark.LEFT_EYE].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE].y)
                    right_eye = (landmarks[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y)
                    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                except IndexError:
                    # Certains points ne sont pas détectés, on ignore cette frame
                    continue

                cal_left = distance_2d(left_shoulder, left_hip)
                cal_right = distance_2d(right_shoulder, right_hip)
                cal_chin = distance_2d(nose, right_shoulder)
                cv2.putText(image, 'Prendre une posture correcte', (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                

                # Affichage des résultats
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


            cv2.imshow('Detection Posture Assise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i += 1
    cap.release()
    cv2.destroyAllWindows()
    return cal_left,cal_right, cal_chin

if __name__ == "__main__":
    cal_left = 0
    cal_right = 0
    cal_chin = 0
    cal_left, cal_right, cal_chin = calibration(cal_left,cal_right, cal_chin)
    main(cal_left, cal_right, cal_chin)
