import websocket
import threading
import time

# Configuration du serveur WebSocket (adapte l'IP/port de ton ESP32)
WS_URL = "ws://10.1.224.71:80"  # Exemple : remplace par l'IP réelle de l'ESP32

# Variable globale pour la connexion WebSocket
ws = None
connected = False

def on_open(ws):
    global connected
    connected = True
    print("[WebSocket] Connexion établie avec le serveur ESP32.")

def on_close(ws, close_status_code, close_msg):
    global connected
    connected = False
    print("[WebSocket] Connexion fermée.")

def on_error(ws, error):
    global connected
    connected = False
    print(f"[WebSocket] Erreur : {error}")

def connect_websocket():
    """Fonction pour établir la connexion WebSocket dans un thread."""
    global ws
    while True:
        try:
            ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_close=on_close, on_error=on_error)
            ws.run_forever()
        except Exception as e:
            print(f"[WebSocket] Échec de connexion, retry dans 5s : {e}")
            time.sleep(5)

def send_message(message):
    """Fonction pour envoyer un message au serveur WebSocket.
    Appelle-la depuis main.py pour envoyer des données."""
    global ws, connected
    if connected and ws:
        try:
            ws.send(message)
            print(f"[WebSocket] Message envoyé : {message}")
        except Exception as e:
            print(f"[WebSocket] Erreur envoi : {e}")
    else:
        print(f"[WebSocket] Non connecté, message ignoré : {message}")

# Démarre le thread de connexion au WebSocket au chargement du module
ws_thread = threading.Thread(target=connect_websocket, daemon=True)
ws_thread.start()