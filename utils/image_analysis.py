"""
Image analysis utilities using MediaPipe, OpenCV, and NumPy
for smile analysis metrics including landmarks, symmetry, and color vitality
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional


class SmileAnalyzer:
    """Analyzes smile images using MediaPipe face mesh and OpenCV"""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Key facial landmark indices for smile analysis
        # These are specific indices from MediaPipe Face Mesh (468 landmarks)
        # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        self.LEFT_MOUTH_CORNER = 61
        self.RIGHT_MOUTH_CORNER = 291
        self.UPPER_LIP_CENTER = 13  # Top of upper lip center
        self.LOWER_LIP_CENTER = 14  # Bottom of lower lip center
        self.UPPER_LIP_TOP = 0  # Vermillion border top
        self.LOWER_LIP_BOTTOM = 17  # Vermillion border bottom
        
    def analyze_image(self, image_array: np.ndarray) -> Dict[str, float]:
        """
        Analyze smile image and return metrics
        
        Args:
            image_array: NumPy array of the image in BGR format
            
        Returns:
            Dictionary containing alignment, symmetry, and color_vitality scores
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Get facial landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                raise ValueError("No face detected in the image")
            
            # Extract landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image_array.shape[:2]
            
            # Convert normalized landmarks to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Calculate metrics
            alignment = self._calculate_alignment(landmarks, w, h)
            symmetry = self._calculate_symmetry(landmarks, w, h)
            color_vitality = self._calculate_color_vitality(image_array, landmarks)
            
            return {
                'alignment': alignment,
                'symmetry': symmetry,
                'color_vitality': color_vitality
            }
            
        except Exception as e:
            raise Exception(f"Analysis error: {str(e)}")
    
    def _calculate_alignment(self, landmarks: list, width: int, height: int) -> float:
        """
        Calculate smile alignment score based on mouth corner positions
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            width: Image width
            height: Image height
            
        Returns:
            Alignment score from 0-10
        """
        try:
            # Get mouth corner landmarks
            left_corner = landmarks[self.LEFT_MOUTH_CORNER]
            right_corner = landmarks[self.RIGHT_MOUTH_CORNER]
            
            # Calculate horizontal alignment
            y_diff = abs(left_corner[1] - right_corner[1])
            mouth_width = abs(right_corner[0] - left_corner[0])
            
            # Normalize by mouth width to handle different image sizes
            alignment_ratio = 1 - min(y_diff / max(mouth_width, 1), 1.0)
            
            # Calculate center alignment (how centered the mouth is)
            mouth_center_x = (left_corner[0] + right_corner[0]) / 2
            face_center_x = width / 2
            center_offset = abs(mouth_center_x - face_center_x) / (width / 2)
            center_alignment = 1 - min(center_offset, 1.0)
            
            # Combine horizontal and center alignment
            alignment_score = (alignment_ratio * 0.7 + center_alignment * 0.3) * 10
            
            return round(max(min(alignment_score, 10.0), 0.0), 2)
            
        except Exception as e:
            print(f"Alignment calculation error: {e}")
            return 5.0
    
    def _calculate_symmetry(self, landmarks: list, width: int, height: int) -> float:
        """
        Calculate facial symmetry score
        
        Args:
            landmarks: List of (x, y) landmark coordinates
            width: Image width
            height: Image height
            
        Returns:
            Symmetry score from 0-10
        """
        try:
            # Calculate face center line
            face_center_x = width / 2
            
            # Analyze left and right side distances from center
            left_distances = []
            right_distances = []
            
            # Sample key landmarks on left and right sides
            # MediaPipe landmarks are roughly symmetric
            for i in range(len(landmarks)):
                x, y = landmarks[i]
                
                # Left side of face
                if x < face_center_x:
                    left_distances.append(abs(x - face_center_x))
                # Right side of face
                elif x > face_center_x:
                    right_distances.append(abs(x - face_center_x))
            
            if not left_distances or not right_distances:
                return 5.0
            
            # Calculate average distances
            avg_left = np.mean(left_distances)
            avg_right = np.mean(right_distances)
            
            # Calculate symmetry ratio
            symmetry_ratio = 1 - abs(avg_left - avg_right) / max(avg_left, avg_right, 1)
            
            # Calculate variance in landmark distribution
            left_variance = np.var(left_distances) if len(left_distances) > 1 else 0
            right_variance = np.var(right_distances) if len(right_distances) > 1 else 0
            
            # Combine metrics
            variance_factor = 1 - min(abs(left_variance - right_variance) / max(left_variance + right_variance, 1), 1.0)
            
            symmetry_score = (symmetry_ratio * 0.6 + variance_factor * 0.4) * 10
            
            return round(max(min(symmetry_score, 10.0), 0.0), 2)
            
        except Exception as e:
            print(f"Symmetry calculation error: {e}")
            return 5.0
    
    def _calculate_color_vitality(self, image: np.ndarray, landmarks: list) -> float:
        """
        Calculate color vitality score based on brightness, contrast, and color analysis
        
        Args:
            image: NumPy array of the image
            landmarks: List of (x, y) landmark coordinates
            
        Returns:
            Color vitality score from 0-10
        """
        try:
            # Extract mouth region using landmarks
            mouth_points = [landmarks[i] for i in [self.LEFT_MOUTH_CORNER, 
                                                   self.RIGHT_MOUTH_CORNER, 
                                                   self.UPPER_LIP_TOP, 
                                                   self.LOWER_LIP_BOTTOM]]
            
            if not mouth_points:
                return 5.0
            
            # Create bounding box around mouth
            xs = [p[0] for p in mouth_points]
            ys = [p[1] for p in mouth_points]
            
            x_min, x_max = max(0, min(xs) - 20), min(image.shape[1], max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(image.shape[0], max(ys) + 20)
            
            mouth_region = image[y_min:y_max, x_min:x_max]
            
            if mouth_region.size == 0:
                return 5.0
            
            # Convert to different color spaces for analysis
            mouth_hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            mouth_lab = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2LAB)
            
            # Calculate brightness (V channel in HSV)
            brightness = np.mean(mouth_hsv[:, :, 2])
            brightness_score = min(brightness / 255.0, 1.0)
            
            # Calculate saturation (S channel in HSV)
            saturation = np.mean(mouth_hsv[:, :, 1])
            saturation_score = min(saturation / 255.0, 1.0)
            
            # Calculate L* (lightness) from LAB
            lightness = np.mean(mouth_lab[:, :, 0])
            lightness_score = min(lightness / 255.0, 1.0)
            
            # Calculate color contrast
            std_dev = np.std(mouth_region)
            contrast_score = min(std_dev / 50.0, 1.0)  # Normalize to 0-1
            
            # Combine metrics with weights
            vitality_score = (
                brightness_score * 0.3 +
                saturation_score * 0.25 +
                lightness_score * 0.25 +
                contrast_score * 0.2
            ) * 10
            
            return round(max(min(vitality_score, 10.0), 0.0), 2)
            
        except Exception as e:
            print(f"Color vitality calculation error: {e}")
            return 5.0
    
    def close(self):
        """Clean up MediaPipe resources"""
        if self.face_mesh:
            self.face_mesh.close()


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        NumPy array of the image in BGR format
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


def validate_image(image_array: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate image for processing
    
    Args:
        image_array: NumPy array of the image
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image_array is None or image_array.size == 0:
        return False, "Invalid or empty image"
    
    h, w = image_array.shape[:2]
    
    if h < 100 or w < 100:
        return False, "Image is too small. Minimum size is 100x100 pixels"
    
    if h > 5000 or w > 5000:
        return False, "Image is too large. Maximum size is 5000x5000 pixels"
    
    return True, None
