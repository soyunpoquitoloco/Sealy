"""
Main entry point for Fatigue Detection System.
Real-time fatigue detection using webcam.
"""

import cv2
import time
import argparse
import logging
import sys
from typing import Optional

from config import Config
from fatigue_detector import FatigueDetector


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time Fatigue Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default settings
  %(prog)s --camera-index 1         # Use camera 1
  %(prog)s --no-calibration         # Skip calibration
  %(prog)s --ear-threshold 0.20     # Set custom EAR threshold
        """
    )
    
    parser.add_argument(
        '--camera-index',
        type=int,
        default=Config.CAMERA_INDEX,
        help=f'Camera index (default: {Config.CAMERA_INDEX})'
    )
    
    parser.add_argument(
        '--no-calibration',
        action='store_true',
        help='Skip calibration and use default thresholds'
    )
    
    parser.add_argument(
        '--ear-threshold',
        type=float,
        default=Config.EAR_THRESHOLD,
        help=f'EAR threshold for closed eyes (default: {Config.EAR_THRESHOLD})'
    )
    
    parser.add_argument(
        '--mar-threshold',
        type=float,
        default=Config.MAR_THRESHOLD,
        help=f'MAR threshold for yawn detection (default: {Config.MAR_THRESHOLD})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def draw_ui(frame, metrics: dict, fps: float, calibration_progress: Optional[float] = None):
    """
    Draw UI overlays on frame.
    
    Args:
        frame: Image frame
        metrics: Dictionary of metrics
        fps: Current FPS
        calibration_progress: Calibration progress (0-1) if in calibration mode
    """
    h, w = frame.shape[:2]
    
    # Determine status color
    if metrics['fatigue_detected']:
        status_color = Config.ALERT_COLOR
        status_text = "FATIGUE DETECTED!"
    elif metrics.get('eyes_closed_too_long', False):
        status_color = Config.WARNING_COLOR
        status_text = "WARNING - EYES CLOSED TOO LONG"
    elif metrics.get('hand_near_mouth', False):
        status_color = Config.WARNING_COLOR
        status_text = "WARNING - YAWN DETECTED"
    elif metrics['fatigue_score'] > Config.FATIGUE_SCORE_THRESHOLD * 0.7:
        status_color = Config.WARNING_COLOR
        status_text = "WARNING"
    else:
        status_color = Config.OK_COLOR
        status_text = "OK"
    
    # Draw status banner if fatigue detected
    if metrics['fatigue_detected']:
        cv2.rectangle(frame, (0, 0), (w, 60), Config.ALERT_COLOR, -1)
        cv2.putText(
            frame, status_text, (w//2 - 150, 40),
            Config.FONT, 1.2, (255, 255, 255), 3
        )
    
    # Draw calibration progress
    if calibration_progress is not None:
        progress_text = f"CALIBRATION: {calibration_progress*100:.0f}%"
        cv2.rectangle(frame, (0, h-80), (w, h), (50, 50, 50), -1)
        cv2.putText(
            frame, progress_text, (w//2 - 120, h - 50),
            Config.FONT, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Keep your eyes open and look at the screen", (w//2 - 250, h - 20),
            Config.FONT, 0.6, (200, 200, 200), 1
        )
        
        # Progress bar
        bar_width = int((w - 40) * calibration_progress)
        cv2.rectangle(frame, (20, h - 90), (20 + bar_width, h - 85), Config.OK_COLOR, -1)
        cv2.rectangle(frame, (20, h - 90), (w - 20, h - 85), (255, 255, 255), 2)
    
    # Draw metrics panel (top-left)
    y_offset = 70 if metrics['fatigue_detected'] else 20
    line_height = 25
    
    metrics_to_display = [
        f"FPS: {fps:.1f}",
        f"Status: {status_text}",
        f"EAR: {metrics['ear_avg']:.3f}",
        f"MAR: {metrics['mar']:.3f}",
        f"PERCLOS: {metrics['perclos']*100:.1f}%",
        f"Yawns/min: {metrics['yawns_per_min']:.1f}",
        f"Hand/mouth: {metrics.get('hand_mouth_rate', 0.0):.1f}/min",
        f"Score: {metrics['fatigue_score']:.2f}",
    ]
    
    # Add hand indicator if detected
    if metrics.get('hand_near_mouth', False):
        metrics_to_display.insert(4, "HAND NEAR MOUTH!")
    
    # Draw background for metrics
    panel_height = len(metrics_to_display) * line_height + 20
    cv2.rectangle(frame, (10, y_offset), (280, y_offset + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, y_offset), (280, y_offset + panel_height), (255, 255, 255), 1)
    
    # Draw metrics text
    for i, text in enumerate(metrics_to_display):
        y_pos = y_offset + 20 + i * line_height
        # Highlight hand detection and status
        if "HAND NEAR MOUTH" in text:
            color = Config.WARNING_COLOR
        elif i == 1:
            color = status_color
        else:
            color = Config.TEXT_COLOR
        cv2.putText(
            frame, text, (20, y_pos),
            Config.FONT, Config.FONT_SCALE, color, Config.FONT_THICKNESS
        )
    
    # Draw face detection status
    if not metrics['face_detected']:
        cv2.putText(
            frame, "NO FACE DETECTED", (w//2 - 120, h//2),
            Config.FONT, 0.8, Config.ALERT_COLOR, 2
        )
    
    # Draw instructions
    instructions = "Press 'q' to quit | 'r' to reset"
    cv2.putText(
        frame, instructions, (w - 350, h - 10),
        Config.FONT, 0.5, Config.TEXT_COLOR, 1
    )


def main():
    """Main application loop."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Update configuration
    Config.CAMERA_INDEX = args.camera_index
    Config.EAR_THRESHOLD = args.ear_threshold
    Config.MAR_THRESHOLD = args.mar_threshold
    Config.CALIBRATION_ENABLED = not args.no_calibration
    
    logger.info("Starting Fatigue Detection System")
    logger.info(f"Camera index: {Config.CAMERA_INDEX}")
    logger.info(f"Calibration: {'Enabled' if Config.CALIBRATION_ENABLED else 'Disabled'}")
    
    # Initialize camera
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {Config.CAMERA_INDEX}")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)
    
    # Initialize detector
    detector = FatigueDetector(Config)
    
    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    
    # Calibration state
    calibration_complete = not Config.CALIBRATION_ENABLED
    calibration_start_time = time.time()
    
    logger.info("Application started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            current_time = time.time()
            
            # Calibration phase
            if Config.CALIBRATION_ENABLED and not calibration_complete:
                # Process frame for calibration
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w = frame.shape[:2]
                    
                    # Calculate EAR for calibration
                    from metrics import extract_landmarks
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
                    
                    left_ear = detector._calculate_ear_from_points(left_eye_lms)
                    right_ear = detector._calculate_ear_from_points(right_eye_lms)
                    
                    calibration_complete = detector.calibrate(left_ear, right_ear, current_time)
                
                # Draw calibration UI
                calibration_elapsed = current_time - calibration_start_time
                calibration_progress = min(calibration_elapsed / Config.CALIBRATION_DURATION_SECONDS, 1.0)
                
                metrics = {
                    'face_detected': results.multi_face_landmarks is not None if 'results' in locals() else False,
                    'ear_avg': 0.0, 'mar': 0.0, 'perclos': 0.0,
                    'yawns_per_min': 0.0, 'long_blinks_per_min': 0.0,
                    'fatigue_score': 0.0, 'fatigue_detected': False
                }
                
                draw_ui(frame, metrics, fps, calibration_progress)
                
                if calibration_complete:
                    logger.info("Calibration complete. Starting detection.")
            else:
                # Normal detection
                frame, metrics = detector.process_frame(frame, current_time)
                draw_ui(frame, metrics, fps)
            
            # Calculate FPS
            frame_count += 1
            elapsed = current_time - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = current_time
            
            # Display frame
            cv2.imshow('Fatigue Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('r'):
                logger.info("Reset requested")
                # Reset detector state
                detector.fatigue_detected = False
                detector.fatigue_start_time = 0.0
                detector.last_alert_time = 0.0
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        logger.info("Application terminated")


if __name__ == '__main__':
    main()
