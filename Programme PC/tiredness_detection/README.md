# Fatigue Detection System

Système de détection de fatigue en temps réel utilisant la webcam. Détecte les signes de fatigue comme les yeux fermés, les bâillements et les clignements prolongés.

## Fonctionnalités

- **Détection en temps réel** : Analyse vidéo à 15-30 FPS sur CPU
- **Détection des yeux fermés** : EAR (Eye Aspect Ratio) + PERCLOS
- **Détection des bâillements** : MAR (Mouth Aspect Ratio)
- **Clignements longs** : Détection des micro-siestes
- **Score de fatigue** : Calcul pondéré avec lissage
- **Alertes intelligentes** : Hystérésis + cooldown
- **Calibration automatique** : Adaptation à l'utilisateur
- **Privacy-first** : Tout en local, pas d'enregistrement

## Prérequis

- Python 3.10 ou supérieur
- Webcam fonctionnelle
- macOS, Linux ou Windows

## Installation

### 1. Cloner ou créer le projet

```bash
cd /Users/emilie/tiredness_detection
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancement basique

```bash
python main.py
```

### Options disponibles

```bash
# Utiliser une autre caméra
python main.py --camera-index 1

# Désactiver la calibration
python main.py --no-calibration

# Ajuster les seuils
python main.py --ear-threshold 0.18 --mar-threshold 0.65

# Mode debug
python main.py --log-level DEBUG
```

### Commandes pendant l'exécution

- **`q`** : Quitter l'application
- **`r`** : Réinitialiser l'état de fatigue

## Métriques affichées

| Métrique | Description | Valeurs normales |
|----------|-------------|------------------|
| **EAR** | Eye Aspect Ratio | 0.25-0.35 (ouvert), <0.20 (fermé) |
| **MAR** | Mouth Aspect Ratio | <0.5 (fermé), >0.6 (bâillement) |
| **PERCLOS** | % temps yeux fermés | <15% (OK), >15% (fatigue) |
| **Yawns/min** | Bâillements par minute | <3 (OK), >3 (fatigue) |
| **Score** | Score de fatigue global | <0.5 (OK), >0.5 (fatigue) |

## Configuration

Tous les paramètres sont ajustables dans `config.py` :

### Seuils principaux

```python
# Yeux
EAR_THRESHOLD = 0.21              # Seuil EAR pour yeux fermés
EYE_CLOSED_MIN_FRAMES = 2         # Frames min pour considérer fermé
PERCLOS_THRESHOLD = 0.15          # 15% du temps

# Bâillements
MAR_THRESHOLD = 0.6               # Seuil MAR pour bâillement
YAWN_MIN_FRAMES = 10              # Frames min pour bâillement

# Score de fatigue
FATIGUE_SCORE_THRESHOLD = 0.5     # Seuil d'alerte
FATIGUE_TRIGGER_SECONDS = 3.0     # Durée avant alerte
FATIGUE_COOLDOWN_SECONDS = 10.0   # Cooldown entre alertes
```

### Poids du score de fatigue

```python
WEIGHT_PERCLOS = 0.4       # 40% du score
WEIGHT_YAWNS = 0.3         # 30% du score
WEIGHT_LONG_BLINKS = 0.3   # 30% du score
```

### Calibration

```python
CALIBRATION_ENABLED = True
CALIBRATION_DURATION_SECONDS = 5.0
CLOSED_RATIO = 0.70  # Seuil = 70% de l'EAR moyen ouvert
```

## Architecture

```
tiredness_detection/
├── main.py                 # Point d'entrée, boucle principale
├── fatigue_detector.py     # Classe FatigueDetector
├── metrics.py              # Calculs EAR, MAR, PERCLOS
├── config.py               # Configuration et seuils
├── requirements.txt        # Dépendances
└── README.md              # Documentation
```

### Flux de traitement

```
Webcam → MediaPipe FaceMesh → Landmarks
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
                 Yeux (EAR)                 Bouche (MAR)
                    ↓                           ↓
              ┌─────┴────┐                 Bâillements
              ↓          ↓                      ↓
         PERCLOS    Long Blinks           Yawns/min
              ↓          ↓                      ↓
              └──────────┴──────────────────────┘
                             ↓
                      Fatigue Score
                             ↓
                      ┌──────┴──────┐
                      ↓             ↓
                  OK/WARNING    FATIGUE ALERT
```

## Ajustements et optimisation

### Si trop d'alertes

1. Augmenter `FATIGUE_SCORE_THRESHOLD` (ex: 0.6 ou 0.7)
2. Augmenter `FATIGUE_TRIGGER_SECONDS` (ex: 5.0)
3. Réduire les poids `WEIGHT_*`
4. Augmenter `EAR_THRESHOLD` (ex: 0.19)

### Si pas assez d'alertes

1. Diminuer `FATIGUE_SCORE_THRESHOLD` (ex: 0.4)
2. Diminuer `EAR_THRESHOLD` (ex: 0.23)
3. Diminuer `MAR_THRESHOLD` (ex: 0.55)
4. Augmenter les poids `WEIGHT_*`

### Optimisation performance

- Réduire `FRAME_WIDTH` et `FRAME_HEIGHT` dans config.py
- Désactiver `DRAW_LANDMARKS` si non nécessaire
- Réduire `PERCLOS_WINDOW_SECONDS` pour moins d'historique

## Résolution de problèmes

### Caméra ne s'ouvre pas

```bash
# Tester les indices disponibles
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Utiliser l'indice trouvé
python main.py --camera-index <indice>
```

### FPS faible (<15)

- Fermer les autres applications utilisant la caméra
- Réduire la résolution dans config.py
- Vérifier les processus CPU

### Pas de détection de visage

- Assurez-vous d'être bien éclairé
- Positionnez-vous face à la caméra
- Ajustez `MIN_DETECTION_CONFIDENCE` dans config.py

### Alertes erratiques

- Lancez avec calibration activée
- Ajustez `SCORE_SMOOTHING_ALPHA` (0.2-0.4)
- Augmentez `FATIGUE_TRIGGER_SECONDS`

## Landmarks MediaPipe

Le système utilise MediaPipe FaceMesh (468 landmarks) :

- **Yeux gauche** : indices 33, 160, 158, 133, 153, 144, etc.
- **Yeux droit** : indices 362, 385, 387, 263, 373, 380, etc.
- **Bouche** : indices 61, 291, 13, 14, 12, 15, 11, 16, etc.

Voir [MediaPipe Face Mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md) pour les détails.

## Améliorations futures (TODO)

- [ ] **Head pose estimation** : Détection de la tête qui tombe
- [ ] **Export des données** : Logging des métriques (CSV)
- [ ] **Alertes sonores** : Notification audio configurable
- [ ] **Dashboard** : Graphiques temps réel
- [ ] **Multi-visages** : Support de plusieurs personnes
- [ ] **GPU acceleration** : Support CUDA/Metal
- [ ] **Configuration UI** : Interface pour ajuster les seuils

## Licence

Projet éducatif - Libre d'utilisation et de modification.

## Crédits

- **MediaPipe** : Google (détection de visage et landmarks)
- **OpenCV** : Traitement d'image et vidéo
- **EAR/MAR** : Métriques basées sur la recherche académique

## Support

Pour toute question ou problème, ouvrez une issue ou consultez les logs avec `--log-level DEBUG`.

---

**Note** : Ce système est conçu pour la détection de fatigue en temps réel mais ne remplace pas une évaluation médicale professionnelle. Utilisez-le de manière responsable.
