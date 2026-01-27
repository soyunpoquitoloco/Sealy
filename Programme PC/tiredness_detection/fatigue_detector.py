"""
Fatigue Detector module.
Main class for detecting fatigue signs from facial landmarks.
"""

import time
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
import mediapipe as mp

from config import Config
from metrics import (
    calculate_ear, calculate_mar, calculate_perclos,
    calculate_yawn_rate, calculate_long_blink_rate,
    exponential_moving_average, extract_landmarks,
    draw_landmarks_on_image, normalize_value
)


class FatigueDetector:
    """
    Main fatigue detection class.
    Processes facial landmarks and detects signs of fatigue.
    """
    
    def __init__(self, config: Config = Config):
        """
        Initialize the fatigue detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = None
        if config.HAND_DETECTION_ENABLED:
            self.hands = self.mp_hands.Hands(
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.HAND_MIN_TRACKING_CONFIDENCE
            )
            self.logger.info("Hand detection enabled")
        
        # State variables
        self.ear_threshold = config.EAR_THRESHOLD
        self.calibrated = False
        
        # History tracking
        self.ear_history: List[float] = []
        self.mar_history: List[float] = []
        self.eye_closed_history: List[bool] = []
        self.timestamps: List[float] = []
        
        # Event tracking
        self.yawn_timestamps: List[float] = []
        self.long_blink_timestamps: List[float] = []
        self.hand_mouth_timestamps: List[float] = []
        
        # Current state
        self.eyes_closed_frames = 0
        self.mouth_open_frames = 0
        self.last_yawn_time = 0.0
        self.in_yawn = False
        self.in_blink = False
        self.blink_start_time = 0.0
        self.hand_near_mouth = False
        self.hand_mouth_frames = 0
        self.hand_mouth_start_time = 0.0
        self.eyes_closed_too_long = False
        self.eyes_closed_start_time = 0.0
        
        # Fatigue score tracking
        self.fatigue_score = 0.0
        self.fatigue_score_ema = 0.0
        self.fatigue_detected = False
        self.fatigue_start_time = 0.0
        self.last_alert_time = 0.0
        
        # Calibration
        self.calibration_samples: List[float] = []
        self.calibration_start_time = 0.0
        
        self.logger.info("FatigueDetector initialized")
    
    def calibrate(self, ear_left: float, ear_right: float, timestamp: float) -> bool:
        """
        Calibrate EAR threshold based on user's open eyes.
        
        Args:
            ear_left: Left eye EAR
            ear_right: Right eye EAR
            timestamp: Current timestamp
            
        Returns:
            True if calibration is complete, False otherwise
        """
        if self.calibration_start_time == 0.0:
            self.calibration_start_time = timestamp
            self.logger.info("Calibration started")
        
        # Collect samples
        avg_ear = (ear_left + ear_right) / 2.0
        self.calibration_samples.append(avg_ear)
        
        elapsed = timestamp - self.calibration_start_time
        
        # Check if calibration period is complete
        if elapsed >= self.config.CALIBRATION_DURATION_SECONDS:
            if len(self.calibration_samples) >= self.config.CALIBRATION_MIN_SAMPLES:
                # Calculate calibrated threshold
                mean_ear = np.mean(self.calibration_samples)
                self.ear_threshold = mean_ear * self.config.CLOSED_RATIO
                self.calibrated = True
                self.logger.info(
                    f"Calibration complete. Mean EAR: {mean_ear:.3f}, "
                    f"Threshold: {self.ear_threshold:.3f}"
                )
                return True
            else:
                self.logger.warning(
                    f"Insufficient calibration samples: {len(self.calibration_samples)}"
                )
                self.calibrated = False
                return True
        
        return False
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame and detect fatigue.
        
        Args:
            frame: BGR image frame from webcam
            timestamp: Current timestamp
            
        Returns:
            Tuple of (processed frame with overlays, metrics dictionary)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Process hands if enabled
        hand_results = None
        if self.hands is not None:
            hand_results = self.hands.process(rgb_frame)
        
        metrics = {
            'face_detected': False,
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'mar': 0.0,
            'perclos': 0.0,
            'yawns_per_min': 0.0,
            'long_blinks_per_min': 0.0,
            'fatigue_score': 0.0,
            'fatigue_detected': False,
            'calibrated': self.calibrated,
            'hand_near_mouth': False,
            'hand_mouth_rate': 0.0,
            'eyes_closed_too_long': False
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Extract landmarks
            left_eye_lms = extract_landmarks(
                face_landmarks, 
                [33, 160, 158, 133, 153, 144, 159, 145, 157],
                w, h
            )
            right_eye_lms = extract_landmarks(
                face_landmarks,
                [362, 385, 387, 263, 373, 380, 386, 374],
                w, h
            )
            mouth_lms = extract_landmarks(
                face_landmarks,
                [61, 291, 13, 14, 12, 15, 11, 16],
                w, h
            )
            
            # Calculate EAR for both eyes
            # For left eye: use indices relative to extracted landmarks
            left_ear = self._calculate_ear_from_points(left_eye_lms)
            right_ear = self._calculate_ear_from_points(right_eye_lms)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR
            mar = self._calculate_mar_from_points(mouth_lms)
            
            # Update metrics
            metrics['face_detected'] = True
            metrics['ear_left'] = left_ear
            metrics['ear_right'] = right_ear
            metrics['ear_avg'] = avg_ear
            metrics['mar'] = mar
            
            # Draw landmarks if enabled
            if self.config.DRAW_LANDMARKS:
                draw_landmarks_on_image(frame, left_eye_lms[:6], self.config.LANDMARK_COLOR, 1, True)
                draw_landmarks_on_image(frame, right_eye_lms[:6], self.config.LANDMARK_COLOR, 1, True)
                draw_landmarks_on_image(frame, mouth_lms[:2], self.config.LANDMARK_COLOR, 1, False)
            
            # Detect hand near mouth
            hand_near_mouth = False
            if hand_results and hand_results.multi_hand_landmarks:
                hand_near_mouth = self._detect_hand_near_mouth(
                    hand_results, mouth_lms, w, h, frame
                )
            self._track_hand_near_mouth(hand_near_mouth, timestamp)
            
            # Process fatigue detection
            self._update_history(avg_ear, mar, timestamp)
            self._detect_eye_closure(avg_ear, timestamp)
            self._detect_yawn(mar, timestamp)
            
            # Calculate metrics
            perclos = calculate_perclos(
                self.eye_closed_history,
                self.timestamps,
                self.config.PERCLOS_WINDOW_SECONDS
            )
            yawns_per_min = calculate_yawn_rate(
                self.yawn_timestamps,
                timestamp,
                self.config.YAWN_RATE_WINDOW_SECONDS
            )
            long_blinks_per_min = calculate_long_blink_rate(
                self.long_blink_timestamps,
                timestamp,
                self.config.YAWN_RATE_WINDOW_SECONDS
            )
            hand_mouth_rate = calculate_yawn_rate(
                self.hand_mouth_timestamps,
                timestamp,
                self.config.YAWN_RATE_WINDOW_SECONDS
            )
            
            # Calculate fatigue score
            fatigue_score = self._calculate_fatigue_score(
                perclos, yawns_per_min, long_blinks_per_min, hand_mouth_rate
            )
            
            # Detect fatigue state
            fatigue_detected = self._detect_fatigue(fatigue_score, timestamp)
            
            # Update metrics
            metrics['perclos'] = perclos
            metrics['yawns_per_min'] = yawns_per_min
            metrics['long_blinks_per_min'] = long_blinks_per_min
            metrics['fatigue_score'] = fatigue_score
            metrics['fatigue_detected'] = fatigue_detected
            metrics['hand_near_mouth'] = self.hand_near_mouth
            metrics['hand_mouth_rate'] = hand_mouth_rate
            metrics['eyes_closed_too_long'] = self.eyes_closed_too_long
        
        return frame, metrics
    
    def _calculate_ear_from_points(self, eye_points: np.ndarray) -> float:
        """Calculate EAR from eye landmark points."""
        if len(eye_points) < 6:
            return 0.0
        
        # Vertical distances (assuming specific ordering)
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h < 1e-6:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_mar_from_points(self, mouth_points: np.ndarray) -> float:
        """Calculate MAR from mouth landmark points."""
        if len(mouth_points) < 8:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(mouth_points[2] - mouth_points[3])
        v2 = np.linalg.norm(mouth_points[4] - mouth_points[5])
        v3 = np.linalg.norm(mouth_points[6] - mouth_points[7])
        
        # Horizontal distance
        h = np.linalg.norm(mouth_points[0] - mouth_points[1])
        
        if h < 1e-6:
            return 0.0
        
        mar = (v1 + v2 + v3) / (2.0 * h)
        return mar
    
    def _update_history(self, ear: float, mar: float, timestamp: float):
        """Update history buffers with new measurements."""
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.timestamps.append(timestamp)
        
        # Keep only recent history (last 60 seconds)
        max_history_time = 60.0
        cutoff_time = timestamp - max_history_time
        
        while len(self.timestamps) > 0 and self.timestamps[0] < cutoff_time:
            self.ear_history.pop(0)
            self.mar_history.pop(0)
            self.timestamps.pop(0)
            if len(self.eye_closed_history) > 0:
                self.eye_closed_history.pop(0)
    
    def _detect_eye_closure(self, ear: float, timestamp: float):
        """Detect eye closure and long blinks."""
        eyes_closed = ear < self.ear_threshold
        self.eye_closed_history.append(eyes_closed)
        
        if eyes_closed:
            self.eyes_closed_frames += 1
            
            # Start blink timer
            if not self.in_blink:
                self.in_blink = True
                self.blink_start_time = timestamp
            
            # Start eyes closed timer for warning
            if self.eyes_closed_start_time == 0.0:
                self.eyes_closed_start_time = timestamp
            
            # Check if eyes have been closed too long
            duration = timestamp - self.eyes_closed_start_time
            if duration >= self.config.EYE_CLOSED_WARNING_SECONDS:
                if not self.eyes_closed_too_long:
                    self.eyes_closed_too_long = True
                    self.logger.warning(f"Eyes closed for {duration:.1f}s - WARNING!")
        else:
            # Check if it was a long blink
            if self.in_blink:
                blink_duration_ms = (timestamp - self.blink_start_time) * 1000
                if blink_duration_ms > self.config.LONG_BLINK_THRESHOLD_MS:
                    self.long_blink_timestamps.append(timestamp)
                    self.logger.debug(f"Long blink detected: {blink_duration_ms:.0f}ms")
            
            # Reset immediately when eyes open
            self.eyes_closed_frames = 0
            self.in_blink = False
            self.eyes_closed_too_long = False
            self.eyes_closed_start_time = 0.0
    
    def _detect_yawn(self, mar: float, timestamp: float):
        """Detect yawning."""
        mouth_open = mar > self.config.MAR_THRESHOLD
        
        if mouth_open:
            self.mouth_open_frames += 1
            self.in_yawn = True
        else:
            # Check if it was a yawn
            if self.in_yawn and self.mouth_open_frames >= self.config.YAWN_MIN_FRAMES:
                # Apply cooldown to avoid counting same yawn multiple times
                if timestamp - self.last_yawn_time > (self.config.YAWN_COOLDOWN_FRAMES / self.config.TARGET_FPS):
                    self.yawn_timestamps.append(timestamp)
                    self.last_yawn_time = timestamp
                    self.logger.debug(f"Yawn detected (frames: {self.mouth_open_frames})")
            
            self.mouth_open_frames = 0
            self.in_yawn = False
    
    def _calculate_fatigue_score(self, perclos: float, yawns_per_min: float, 
                                 long_blinks_per_min: float, hand_mouth_rate: float = 0.0) -> float:
        """
        Calculate weighted fatigue score.
        
        Returns:
            Fatigue score (0.0 to 1.0+)
        """
        # Normalize inputs
        perclos_norm = normalize_value(perclos, 0.0, 0.5)  # 0-50% range
        yawns_norm = normalize_value(yawns_per_min, 0.0, 10.0)  # 0-10 yawns/min
        blinks_norm = normalize_value(long_blinks_per_min, 0.0, 10.0)  # 0-10 blinks/min
        hand_mouth_norm = normalize_value(hand_mouth_rate, 0.0, 10.0)  # 0-10 times/min
        
        # Weighted sum
        score = (
            self.config.WEIGHT_PERCLOS * perclos_norm +
            self.config.WEIGHT_YAWNS * yawns_norm +
            self.config.WEIGHT_LONG_BLINKS * blinks_norm +
            self.config.WEIGHT_HAND_MOUTH * hand_mouth_norm
        )
        
        # Apply exponential moving average for smoothing
        self.fatigue_score = score
        self.fatigue_score_ema = exponential_moving_average(
            score,
            self.fatigue_score_ema,
            self.config.SCORE_SMOOTHING_ALPHA
        )
        
        return self.fatigue_score_ema
    
    def _detect_fatigue(self, fatigue_score: float, timestamp: float) -> bool:
        """
        Detect fatigue state with hysteresis and cooldown.
        
        Returns:
            True if fatigue is detected, False otherwise
        """
        # Check cooldown
        if timestamp - self.last_alert_time < self.config.FATIGUE_COOLDOWN_SECONDS:
            return False
        
        # Check if score exceeds threshold
        if fatigue_score > self.config.FATIGUE_SCORE_THRESHOLD:
            if not self.fatigue_detected:
                # Start fatigue timer
                if self.fatigue_start_time == 0.0:
                    self.fatigue_start_time = timestamp
                
                # Check if condition maintained for trigger duration
                if timestamp - self.fatigue_start_time >= self.config.FATIGUE_TRIGGER_SECONDS:
                    self.fatigue_detected = True
                    self.last_alert_time = timestamp
                    self.logger.warning("FATIGUE DETECTED!")
                    return True
        else:
            # Reset if score drops
            self.fatigue_detected = False
            self.fatigue_start_time = 0.0
        
        return self.fatigue_detected
    
    def release(self):
        """Release resources."""
        self.face_mesh.close()
        if self.hands is not None:
            self.hands.close()
        self.logger.info("FatigueDetector released")
    
    def _detect_hand_near_mouth(self, hand_results, mouth_lms, w, h, frame) -> bool:
        """
        Detect if a hand is near the mouth area.
        
        Args:
            hand_results: MediaPipe hand detection results
            mouth_lms: Mouth landmarks (numpy array)
            w: Frame width
            h: Frame height
            frame: Frame to draw on (optional)
            
        Returns:
            True if hand is near mouth
        """
        if len(mouth_lms) < 2:
            return False
        
        # Get mouth center position (normalized)
        mouth_center = np.mean(mouth_lms[:2], axis=0)
        mouth_center_norm = mouth_center / np.array([w, h])
        
        # Check each detected hand
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get palm center (average of landmarks 0, 5, 9, 13, 17)
            palm_indices = [0, 5, 9, 13, 17]
            palm_points = []
            
            for idx in palm_indices:
                landmark = hand_landmarks.landmark[idx]
                palm_points.append([landmark.x, landmark.y])
            
            palm_center = np.mean(palm_points, axis=0)
            
            # Calculate distance between palm and mouth
            distance = np.linalg.norm(palm_center - mouth_center_norm)
            
            # Draw hand landmarks if close to mouth and drawing enabled
            if distance < self.config.HAND_MOUTH_DISTANCE_THRESHOLD:
                if self.config.DRAW_LANDMARKS:
                    # Draw hand landmarks
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
                    
                    # Draw connection line
                    mouth_center_px = (int(mouth_center[0]), int(mouth_center[1]))
                    palm_center_px = (int(palm_center[0] * w), int(palm_center[1] * h))
                    cv2.line(frame, palm_center_px, mouth_center_px, (255, 0, 255), 2)
                
                self.logger.debug(f"Hand near mouth detected (distance: {distance:.3f})")
                return True
        
        return False
    
    def _track_hand_near_mouth(self, hand_near_mouth: bool, timestamp: float):
        """
        Track hand near mouth events.
        
        Args:
            hand_near_mouth: Whether hand is currently near mouth
            timestamp: Current timestamp
        """
        if hand_near_mouth:
            self.hand_mouth_frames += 1
            
            # Start timer if this is the beginning of detection
            if self.hand_mouth_start_time == 0.0:
                self.hand_mouth_start_time = timestamp
            
            # Check if duration exceeds minimum threshold
            duration = timestamp - self.hand_mouth_start_time
            if not self.hand_near_mouth and duration >= self.config.HAND_MOUTH_MIN_DURATION_SECONDS:
                # Log event only after minimum duration to avoid false positives
                self.hand_mouth_timestamps.append(timestamp)
                self.hand_near_mouth = True
                self.logger.debug(f"Hand covering mouth detected (duration: {duration:.1f}s) - possible yawn suppression")
        else:
            # Reset when hand is no longer near mouth
            self.hand_mouth_frames = 0
            self.hand_mouth_start_time = 0.0
            self.hand_near_mouth = False
