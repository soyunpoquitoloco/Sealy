import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os

# Assurez-vous que les fichiers vidéo existent dans le répertoire courant
# Remplacez 'video1.mp4' et 'video2.mp4' par vos propres fichiers vidéo
video_files = ['Start.mkv', 'Start.mkv']

# Ouvrir les vidéos avec OpenCV
caps = []
for video_file in video_files:
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Erreur : Impossible d'ouvrir {video_file}")
            exit(1)
        caps.append(cap)
    else:
        print(f"Erreur : {video_file} non trouvé.")
        exit(1)

# Variables globales
current_video = 0  # 0 pour video1, 1 pour video2
change_pending = False
fps = 24  # FPS approximatif pour le délai (ajustez selon vos vidéos)
delay = int(1000 / fps)  # Délai en ms entre les frames

def animate():
    global current_video, change_pending
    cap = caps[current_video]
    ret, frame = cap.read()
    if ret:
        # Convertir BGR (OpenCV) en RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk  # Garder une référence
        # Planifier le prochain frame
        root.after(delay, animate)
    else:
        # Fin de la vidéo, revenir au début pour boucler
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Si un changement est en attente, basculer la vidéo
        if change_pending:
            current_video = 1 - current_video  # Bascule entre 0 et 1
            change_pending = False
            print("video changée")
        # Continuer l'animation (recommencer la boucle)
        root.after(delay, animate)

def on_key_press(event):
    global change_pending
    if event.char == 'p':
        change_pending = True

# Créer la fenêtre
root = tk.Tk()
root.title("Sealy")
# Créer un label pour afficher les frames
label = tk.Label(root)
label.pack()

# Lier la pression de touche
root.bind('<Key>', on_key_press)

# Démarrer l'animation
animate()

# Lancer la boucle principale
root.mainloop()

# Libérer les ressources à la fin
for cap in caps:
    cap.release()
