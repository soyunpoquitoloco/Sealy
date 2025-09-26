from posture import *  # Idem

from deepface import DeepFace

def main(cal_left, cal_right):
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

            # Analyse de la posture
            posture_avachie = False
            dominant_emotion = "Inconnu"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Récupérer les points clés visibles du haut du corps
                try:
                    nose = (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y)
                    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                except IndexError:
                    # Certains points ne sont pas détectés, on ignore cette frame pour la posture
                    pass
                else:
                    # Calculer l'angle entre la tête (nez), l'épaule gauche et l'épaule droite
                    angle_tete = calculate_angle(left_shoulder, nose, right_shoulder)

                    # Calculer l'angle entre les épaules et les hanches pour détecter un dos courbé
                    angle_dos_gauche = calculate_angle(left_hip, left_shoulder, nose)
                    angle_dos_droite = calculate_angle(right_hip, right_shoulder, nose)

                    # Critères simples pour posture avachie assise
                    length_left_side = distance_2d(left_shoulder, left_hip)
                    length_right_side = distance_2d(right_shoulder, right_hip)

                    if length_left_side < cal_left and length_right_side < cal_right:
                        posture_avachie = True

                    # Affichage des landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Analyse des émotions avec DeepFace (sur l'image BGR)
            try:
                emotion_result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)

                if isinstance(emotion_result, list):
                    if len(emotion_result) == 0:
                        dominant_emotion = "Aucun visage"
                    else:
                        emotion_result = emotion_result[0]
                        dominant_emotion = emotion_result['dominant_emotion']
                else:
                    dominant_emotion = emotion_result['dominant_emotion']

            except Exception as e:
                print("Erreur lors de l'analyse des émotions :", e)
                dominant_emotion = "Erreur"

            # Affichage des résultats de posture
            if posture_avachie:
                cv2.putText(image, 'Posture Avachie Detectee', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(image, 'Posture Assise Correcte', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Affichage des résultats d'émotion
            cv2.putText(image, f'Emotion: {dominant_emotion}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Detection Posture + Emotion', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cal_left, cal_right = calibration()
    main(cal_left, cal_right)
