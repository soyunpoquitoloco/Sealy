import cv2
import mediapipe as mp
import math

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

def main(cal_chin):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

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
                try:
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                except IndexError:
                    continue

                length_chin = distance_2d(nose, right_shoulder)
                chin_change = (length_chin - cal_chin) / cal_chin * 100
                posture_avachie = chin_change < -10

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if posture_avachie:
                    cv2.putText(image, 'Posture Avachie Detectee', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(image, 'Posture Assise Correcte', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('Detection Posture Assise', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cal_chin = calibration()
    main(cal_chin)
