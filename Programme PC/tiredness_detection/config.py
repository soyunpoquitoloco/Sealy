"""
Configuration file for Fatigue Detection System
Contains all thresholds, parameters, and settings.
"""

from typing import Dict, Any


class Config:
    """Configuration class for fatigue detection parameters."""
    
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = True
    
    # MediaPipe Hands settings
    HAND_DETECTION_ENABLED: bool = True
    MAX_NUM_HANDS: int = 2
    HAND_MIN_DETECTION_CONFIDENCE: float = 0.5
    HAND_MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Hand near mouth detection
    HAND_MOUTH_DISTANCE_THRESHOLD: float = 0.15  # Normalized distance (0-1)
    HAND_MOUTH_MIN_DURATION_SECONDS: float = 2.0  # Minimum duration to count as yawn
    HAND_YAWN_WEIGHT: float = 0.2  # Weight for hand-over-mouth in fatigue score
    
    # Eye landmarks indices for MediaPipe FaceMesh (468 landmarks model)
    # Left eye: vertical points
    LEFT_EYE_INDICES: list = [
        # Upper eyelid: 159, 158, 157, 173
        # Lower eyelid: 145, 144, 163, 7
        # Corners: 33 (inner), 133 (outer)
        33, 160, 158, 133, 153, 144  # Simplified contour
    ]
    
    RIGHT_EYE_INDICES: list = [
        # Right eye (mirror of left)
        362, 385, 387, 263, 373, 380  # Simplified contour
    ]
    
    # Specific indices for EAR calculation (vertical points)
    LEFT_EYE_VERTICAL: list = [
        [159, 145],  # Upper to lower (left side)
        [158, 153],  # Upper to lower (middle-left)
        [157, 144],  # Upper to lower (middle-right)
    ]
    LEFT_EYE_HORIZONTAL: list = [33, 133]  # Inner to outer corner
    
    RIGHT_EYE_VERTICAL: list = [
        [386, 374],  # Upper to lower (left side)
        [385, 380],  # Upper to lower (middle-left)
        [387, 373],  # Upper to lower (middle-right)
    ]
    RIGHT_EYE_HORIZONTAL: list = [362, 263]  # Inner to outer corner
    
    # Mouth landmarks indices
    MOUTH_INDICES: list = [
        # Outer lips
        61, 291, 0, 17, 269, 405, 314, 17, 84, 181, 91, 146
    ]
    
    # Specific indices for MAR calculation
    MOUTH_VERTICAL: list = [
        [13, 14],    # Upper to lower lip (center)
        [12, 15],    # Upper to lower lip (left of center)
        [11, 16],    # Upper to lower lip (right of center)
    ]
    MOUTH_HORIZONTAL: list = [61, 291]  # Left to right corner
    
    # EAR (Eye Aspect Ratio) parameters
    EAR_THRESHOLD: float = 0.21  # Default threshold, will be calibrated
    EYE_CLOSED_MIN_FRAMES: int = 2  # Minimum frames to consider eyes closed
    EYE_CLOSED_WARNING_SECONDS: float = 2.0  # Warning if eyes closed for this duration
    CLOSED_RATIO: float = 0.70  # For calibration: closed_threshold = open_mean * ratio
    
    # MAR (Mouth Aspect Ratio) parameters
    MAR_THRESHOLD: float = 0.6  # Threshold for yawn detection
    YAWN_MIN_FRAMES: int = 10  # Minimum frames (~333ms at 30fps) for yawn
    YAWN_MIN_MS: int = 300  # Minimum duration in milliseconds
    YAWN_COOLDOWN_FRAMES: int = 15  # Cooldown between yawns
    
    # PERCLOS (Percentage of Eye Closure) parameters
    PERCLOS_WINDOW_SECONDS: float = 30.0  # Window for PERCLOS calculation
    PERCLOS_THRESHOLD: float = 0.15  # 15% of time eyes closed indicates fatigue
    
    # Long blink detection
    LONG_BLINK_THRESHOLD_MS: int = 500  # Blinks longer than 500ms
    LONG_BLINK_THRESHOLD_FRAMES: int = 15  # ~500ms at 30fps
    
    # Fatigue score calculation (weights)
    WEIGHT_PERCLOS: float = 0.3
    WEIGHT_YAWNS: float = 0.25
    WEIGHT_LONG_BLINKS: float = 0.25
    WEIGHT_HAND_MOUTH: float = 0.2  # Hand covering mouth (yawn suppression)
    WEIGHT_HEAD_POSE: float = 0.0  # Not implemented yet
    
    # Fatigue detection parameters
    FATIGUE_SCORE_THRESHOLD: float = 0.5  # Score above this = fatigue
    FATIGUE_TRIGGER_SECONDS: float = 3.0  # Score must be high for this duration
    FATIGUE_COOLDOWN_SECONDS: float = 10.0  # Cooldown after alert
    SCORE_SMOOTHING_ALPHA: float = 0.3  # Exponential moving average factor
    
    # Yawn rate calculation
    YAWN_RATE_WINDOW_SECONDS: float = 60.0  # Calculate yawns per minute
    YAWN_RATE_THRESHOLD: float = 3.0  # 3+ yawns/min indicates fatigue
    
    # Calibration parameters
    CALIBRATION_ENABLED: bool = True
    CALIBRATION_DURATION_SECONDS: float = 5.0
    CALIBRATION_MIN_SAMPLES: int = 30  # Minimum samples for valid calibration
    
    # UI parameters
    FONT: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.6
    FONT_THICKNESS: int = 2
    TEXT_COLOR: tuple = (255, 255, 255)  # White
    ALERT_COLOR: tuple = (0, 0, 255)  # Red
    OK_COLOR: tuple = (0, 255, 0)  # Green
    WARNING_COLOR: tuple = (0, 165, 255)  # Orange
    
    # Drawing parameters
    DRAW_LANDMARKS: bool = True
    LANDMARK_COLOR: tuple = (0, 255, 0)  # Green
    LANDMARK_THICKNESS: int = 1
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
