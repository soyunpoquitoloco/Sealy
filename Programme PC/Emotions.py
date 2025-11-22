import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture vidéo")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            if len(result) == 0:
                print("Aucun visage détecté")
                dominant_emotion = "Inconnu"
            else:
                result = result[0]
                dominant_emotion = result['dominant_emotion']
        else:
            dominant_emotion = result['dominant_emotion']

        cv2.putText(frame, dominant_emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Erreur lors de l'analyse :", e)
        dominant_emotion = "Erreur"

    cv2.imshow("Emotion Detection with DeepFace", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
