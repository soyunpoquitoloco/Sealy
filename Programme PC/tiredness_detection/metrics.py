"""
Metrics module for fatigue detection.
Contains functions to calculate EAR, MAR, PERCLOS, and other metrics.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return np.linalg.norm(point1 - point2)


def calculate_ear(eye_landmarks: np.ndarray, 
                  vertical_indices: List[List[int]], 
                  horizontal_indices: List[int]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR).
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1-p6 are eye landmarks (vertical pairs and horizontal)
    
    Args:
        eye_landmarks: Array of eye landmark coordinates (N, 2)
        vertical_indices: List of pairs of indices for vertical measurements
        horizontal_indices: Pair of indices for horizontal measurement
        
    Returns:
        EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    # Calculate vertical distances
    vertical_distances = []
    for v_pair in vertical_indices:
        p1 = eye_landmarks[v_pair[0]]
        p2 = eye_landmarks[v_pair[1]]
        vertical_distances.append(euclidean_distance(p1, p2))
    
    # Calculate horizontal distance
    p_left = eye_landmarks[horizontal_indices[0]]
    p_right = eye_landmarks[horizontal_indices[1]]
    horizontal_distance = euclidean_distance(p_left, p_right)
    
    # Avoid division by zero
    if horizontal_distance < 1e-6:
        return 0.0
    
    # EAR formula
    ear = sum(vertical_distances) / (2.0 * horizontal_distance)
    return ear


def calculate_mar(mouth_landmarks: np.ndarray,
                  vertical_indices: List[List[int]],
                  horizontal_indices: List[int]) -> float:
    """
    Calculate Mouth Aspect Ratio (MAR).
    
    MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
    Higher values indicate mouth opening (yawning)
    
    Args:
        mouth_landmarks: Array of mouth landmark coordinates (N, 2)
        vertical_indices: List of pairs of indices for vertical measurements
        horizontal_indices: Pair of indices for horizontal measurement
        
    Returns:
        MAR value (typically <0.5 for closed mouth, >0.6 for open/yawn)
    """
    # Calculate vertical distances
    vertical_distances = []
    for v_pair in vertical_indices:
        p1 = mouth_landmarks[v_pair[0]]
        p2 = mouth_landmarks[v_pair[1]]
        vertical_distances.append(euclidean_distance(p1, p2))
    
    # Calculate horizontal distance
    p_left = mouth_landmarks[horizontal_indices[0]]
    p_right = mouth_landmarks[horizontal_indices[1]]
    horizontal_distance = euclidean_distance(p_left, p_right)
    
    # Avoid division by zero
    if horizontal_distance < 1e-6:
        return 0.0
    
    # MAR formula
    mar = sum(vertical_distances) / (2.0 * horizontal_distance)
    return mar


def calculate_perclos(eye_closed_history: List[bool], 
                      timestamps: List[float],
                      window_seconds: float) -> float:
    """
    Calculate PERCLOS (Percentage of Eye Closure).
    
    PERCLOS is the percentage of time in a window where eyes are closed.
    
    Args:
        eye_closed_history: List of boolean values indicating if eyes were closed
        timestamps: Corresponding timestamps for each measurement
        window_seconds: Time window in seconds
        
    Returns:
        PERCLOS value (0.0 to 1.0)
    """
    if len(eye_closed_history) == 0 or len(timestamps) == 0:
        return 0.0
    
    current_time = timestamps[-1]
    window_start = current_time - window_seconds
    
    # Filter data within window
    closed_in_window = []
    for closed, timestamp in zip(eye_closed_history, timestamps):
        if timestamp >= window_start:
            closed_in_window.append(closed)
    
    if len(closed_in_window) == 0:
        return 0.0
    
    # Calculate percentage
    perclos = sum(closed_in_window) / len(closed_in_window)
    return perclos


def calculate_yawn_rate(yawn_timestamps: List[float],
                        current_time: float,
                        window_seconds: float = 60.0) -> float:
    """
    Calculate yawn rate (yawns per minute).
    
    Args:
        yawn_timestamps: List of timestamps when yawns occurred
        current_time: Current timestamp
        window_seconds: Time window in seconds
        
    Returns:
        Yawns per minute
    """
    if len(yawn_timestamps) == 0:
        return 0.0
    
    window_start = current_time - window_seconds
    
    # Count yawns in window
    yawns_in_window = sum(1 for ts in yawn_timestamps if ts >= window_start)
    
    # Convert to yawns per minute
    yawns_per_minute = (yawns_in_window / window_seconds) * 60.0
    return yawns_per_minute


def calculate_long_blink_rate(long_blink_timestamps: List[float],
                               current_time: float,
                               window_seconds: float = 60.0) -> float:
    """
    Calculate long blink rate (long blinks per minute).
    
    Args:
        long_blink_timestamps: List of timestamps when long blinks occurred
        current_time: Current timestamp
        window_seconds: Time window in seconds
        
    Returns:
        Long blinks per minute
    """
    if len(long_blink_timestamps) == 0:
        return 0.0
    
    window_start = current_time - window_seconds
    
    # Count long blinks in window
    long_blinks_in_window = sum(1 for ts in long_blink_timestamps if ts >= window_start)
    
    # Convert to per minute
    long_blinks_per_minute = (long_blinks_in_window / window_seconds) * 60.0
    return long_blinks_per_minute


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value of range
        max_val: Maximum value of range
        
    Returns:
        Normalized value
    """
    if max_val - min_val < 1e-6:
        return 0.0
    return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)


def exponential_moving_average(current_value: float, 
                                previous_ema: float, 
                                alpha: float) -> float:
    """
    Calculate exponential moving average.
    
    EMA = alpha * current + (1 - alpha) * previous
    
    Args:
        current_value: Current value
        previous_ema: Previous EMA value
        alpha: Smoothing factor (0 to 1), higher = more weight to current
        
    Returns:
        Updated EMA value
    """
    return alpha * current_value + (1 - alpha) * previous_ema


def extract_landmarks(face_landmarks, 
                      indices: List[int], 
                      image_width: int, 
                      image_height: int) -> np.ndarray:
    """
    Extract specific landmarks and convert to pixel coordinates.
    
    Args:
        face_landmarks: MediaPipe face landmarks object
        indices: List of landmark indices to extract
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Array of landmark coordinates (N, 2)
    """
    landmarks = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        landmarks.append([x, y])
    
    return np.array(landmarks, dtype=np.float32)


def draw_landmarks_on_image(image: np.ndarray, 
                            landmarks: np.ndarray,
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 1,
                            closed: bool = True) -> None:
    """
    Draw landmarks as a contour on the image.
    
    Args:
        image: Image to draw on
        landmarks: Array of landmark coordinates
        color: BGR color tuple
        thickness: Line thickness
        closed: Whether to close the contour
    """
    points = landmarks.astype(np.int32)
    cv2.polylines(image, [points], closed, color, thickness)
