import cv2
import time
from picamera2 import Picamera2
from deepface import DeepFace

# Initialisation de la caméra avec Picamera2
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    print("Caméra initialisée avec succès.")
except Exception as e:
    print(f"Erreur d'initialisation de la caméra : {e}")
    exit(1)

# Chargeur de classificateur pour détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Erreur : Impossible de charger le classificateur de visages.")
    exit(1)

print("Démarrage de la capture... Appuyez sur Ctrl+C pour arrêter.")
iteration = 0

try:
    while True:
        iteration += 1
        print(f"Iteration {iteration} : Capture en cours...")
        
        # Capture d'une frame
        image = picam2.capture_array()
        print(f"  - Image capturée, taille : {image.shape}, moyenne des pixels : {image.mean():.1f} (doit être >0 si éclairé)")
        
        if image.mean() < 1:  # Si image noire/vide
            print("  - Attention : Image noire ou vide (problème caméra/lumière ?)")
        
        # Détection de visages avec OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"  - Nombre de visages détectés : {len(faces)}")
        
        if len(faces) > 0:
            print("  - Visage(s) trouvé(s) ! Analyse de l'émotion...")
            (x, y, w, h) = faces[0]  # Première face
            face_roi = image[y:y+h, x:x+w]
            
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                dominant_emotion = result[0]['dominant_emotion']
                print(f"  - Émotion dominante : {dominant_emotion}")
                
                # Probabilités
                emotions = result[0]['emotion']
                for emotion, score in emotions.items():
                    if score > 30:  # Seuil abaissé pour plus de détails
                        print(f"    - {emotion}: {score:.1f}%")
                        
            except Exception as e:
                print(f"  - Erreur d'analyse DeepFace : {e}")
        else:
            print("  - Aucun visage détecté. Assurez-vous d'être face à la caméra.")
        
        if iteration % 5 == 0:  # Print périodique pour confirmer que la boucle tourne
            print(f"--- Boucle active (itération {iteration}) ---")
        
        time.sleep(1)  # Pause courte pour debug ; augmentez à 2-3s pour production

except KeyboardInterrupt:
    print("\nArrêt du programme.")
finally:
    picam2.stop()
    print("Caméra arrêtée.")
