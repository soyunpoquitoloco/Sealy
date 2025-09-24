import cv2
import time
from picamera2 import Picamera2
from deepface import DeepFace

# Initialisation de la caméra avec Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

# Chargeur de classificateur pour détection de visages (optionnel)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Démarrage de la capture... Appuyez sur Ctrl+C pour arrêter.")

try:
    while True:
        # Capture d'une frame
        image = picam2.capture_array()
        
        # Détection de visages avec OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Prenez la première face détectée
            (x, y, w, h) = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            print("Analyse de l'émotion...")
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                print(f"Émotion dominante : {dominant_emotion}")
                
                # Probabilités des émotions (seulement les significatives)
                emotions = result[0]['emotion']
                for emotion, score in emotions.items():
                    if score > 50:
                        print(f"  - {emotion}: {score:.1f}%")
                        
            except Exception as e:
                print(f"Erreur d'analyse : {e}")
        
        time.sleep(2)  # Pause pour éviter la surcharge (ajustez à 1-5 secondes)

except KeyboardInterrupt:
    print("\nArrêt du programme.")
finally:
    picam2.stop()
