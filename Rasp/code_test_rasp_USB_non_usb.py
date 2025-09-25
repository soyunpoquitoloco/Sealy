import cv2
import time
import numpy as np
import os
from datetime import datetime
from deepface import DeepFace

# Configuration
USE_USB_CAMERA = True  # True pour USB, False pour Pi Camera
SAVE_ROI = False  # True pour sauvegarder aussi la ROI (visage isolé)
CAPTURES_DIR = "captures"  # Dossier de sauvegarde

# Créer le dossier si absent
os.makedirs(CAPTURES_DIR, exist_ok=True)

if USE_USB_CAMERA:
    # Initialisation caméra USB
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra USB.")
        exit(1)
    print("Caméra USB initialisée avec succès.")
else:
    # Initialisation Pi Camera (Picamera2)
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Caméra Pi initialisée avec succès.")

# Classificateurs : Frontal + Profil
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

if frontal_cascade.empty() or profile_cascade.empty():
    print("Erreur : Impossible de charger les classificateurs de visages.")
    exit(1)

print("Démarrage de la capture... Appuyez sur Ctrl+C pour arrêter.")
iteration = 0
last_capture_time = 0  # Pour éviter captures trop fréquentes (optionnel)

def preprocess_image(gray):
    """Prétraitement pour éclairage faible : Égalisation CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def save_capture(image, faces, face_index=0):
    """Sauvegarde l'image avec rectangle sur le visage"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Microsecondes pour unicité
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(CAPTURES_DIR, filename)
    
    # Dessiner rectangle sur l'image (vert pour frontal, bleu pour profil)
    draw_image = image.copy()
    (x, y, w, h) = faces[face_index]
    color = (0, 255, 0) if face_index < len(frontal_cascade.detectMultiScale(preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))) else (255, 0, 0)
    cv2.rectangle(draw_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(draw_image, f"Visage detecte: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Sauvegarde image complète
    success = cv2.imwrite(filepath, draw_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if success:
        print(f"  - Capture sauvegardée : {filepath}")
    else:
        print(f"  - Erreur sauvegarde : {filepath}")
    
    # Option : Sauvegarde ROI (visage isolé)
    if SAVE_ROI:
        roi_filename = f"roi_{timestamp}.jpg"
        roi_filepath = os.path.join(CAPTURES_DIR, roi_filename)
        roi = image[y:y+h, x:x+w]
        cv2.imwrite(roi_filepath, roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  - ROI sauvegardée : {roi_filepath}")

try:
    while True:
        iteration += 1
        print(f"Iteration {iteration} : Capture en cours...")
        
        if USE_USB_CAMERA:
            ret, image = cap.read()
            if not ret:
                print("  - Erreur de capture USB.")
                time.sleep(1)
                continue
        else:
            image = picam2.capture_array()
        
        print(f"  - Image capturée, taille : {image.shape}, moyenne des pixels : {image.mean():.1f}")
        
        if image.mean() < 5:
            print("  - Image très sombre : Amélioration du contraste appliquée.")
        
        # Conversion grise et prétraitement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = preprocess_image(gray)
        
        # Détection
        frontal_faces = frontal_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        profile_faces = profile_cascade.detectMultiScale(enhanced_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        all_faces = np.vstack((frontal_faces, profile_faces)) if len(frontal_faces) > 0 or len(profile_faces) > 0 else np.array([])
        
        print(f"  - Visages frontaux : {len(frontal_faces)}, profils : {len(profile_faces)}, total : {len(all_faces)}")
        
        if len(all_faces) > 0:
            print("  - Visage(s) trouvé(s) ! Analyse de l'émotion...")
            
            # Sauvegarde à chaque détection (premier visage)
            save_capture(image, all_faces, 0)
            
            (x, y, w, h) = all_faces[0]
            face_roi = image[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (224, 224))  # Pour DeepFace
            
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
                dominant_emotion = result[0]['dominant_emotion']
                print(f"  - Émotion dominante : {dominant_emotion}")
                
                emotions = result[0]['emotion']
                for emotion, score in emotions.items():
                    if score > 20:
                        print(f"    - {emotion}: {score:.1f}%")
                        
            except Exception as e:
                print(f"  - Erreur d'analyse DeepFace : {e}")
                # Fallback
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    dominant_emotion = result[0]['dominant_emotion']
                    print(f"  - Émotion (fallback) : {dominant_emotion}")
                except:
                    print("  - Analyse échouée complètement.")
        else:
            print("  - Aucun visage détecté. Vérifiez distance/angle/éclairage.")
        
        if iteration % 3 == 0:
            print(f"--- Boucle active (itération {iteration}) ---")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\nArrêt du programme.")
finally:
    if USE_USB_CAMERA:
        cap.release()
    else:
        picam2.stop()
    print("Caméra arrêtée.")
    cv2.destroyAllWindows()
