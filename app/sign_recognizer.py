import os
import pickle
import numpy as np

try:
    from app.motion_recognizer import MotionGestureRecognizer
except ImportError:
    from motion_recognizer import MotionGestureRecognizer

try:
    from app.config import get_model_paths, get_current_dataset
except ImportError:
    from config import get_model_paths, get_current_dataset


class SignLanguageRecognizer:
    """ Recognizer for Sign Language letters (ASL/ISL) using MediaPipe hand landmarks. 
    
    For ASL: Single-hand recognition (fingerspelling) - 78 dim features
    For ISL: Unified model handles both 1-hand and 2-hand signs - 162 dim features
             (1-hand features are zero-padded to match 2-hand dimension)
    """

    def __init__(self, dataset: str = None, model_path=None, motion_model_path=None):
        """
        Initialize the recognizer for a specific dataset.

        Args:
            dataset: Dataset name ("ASL" or "ISL"). If None, uses current dataset from config.
            model_path: Optional explicit path to static model file.
            motion_model_path: Optional explicit path to motion model file.
        """
        self.dataset = dataset or get_current_dataset()
        paths = get_model_paths(self.dataset)

        self.ml_model = None          # Unified static model (handles 1-hand ASL, 1/2-hand ISL)
        self.history = []
        self.history_size = 2

        # Determine model paths
        static_model_path = model_path or paths["sign_model"]
        motion_model_path = motion_model_path or paths["motion_model"]

        # Loading unified static ML model
        if os.path.exists(static_model_path):
            try:
                with open(static_model_path, "rb") as f:
                    self.ml_model = pickle.load(f)
                print(f"[SignRecognizer] {self.dataset} static ML model loaded from {static_model_path}")
            except Exception as e:
                print(f"[SignRecognizer] Failed to load static ML model: {e}")

        self.motion_recognizer = MotionGestureRecognizer(dataset=self.dataset, model_path=motion_model_path)

    def switch_dataset(self, dataset: str):
        """Switch to a different dataset and reload models."""
        if dataset == self.dataset:
            return

        self.dataset = dataset
        paths = get_model_paths(dataset)

        # Reload unified static model
        self.ml_model = None
        if os.path.exists(paths["sign_model"]):
            try:
                with open(paths["sign_model"], "rb") as f:
                    self.ml_model = pickle.load(f)
                print(f"[SignRecognizer] {dataset} static ML model loaded from {paths['sign_model']}")
            except Exception as e:
                print(f"[SignRecognizer] Failed to load static ML model: {e}")

        # Reload motion model
        self.motion_recognizer.switch_dataset(dataset)

        # Clear history when switching datasets
        self.history = []

    def extract_features(self, landmarks):
        """
        Extract normalized coordinates, angles, and distances from the 21 MediaPipe landmark points.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Reference point: Wrist (index 0)
        wrist = pts[0]
        pts_norm = pts - wrist

        # Scale normalization based on palm size (0 -> 9 MCP of middle finger)
        palm_size = np.linalg.norm(pts_norm[9])
        if palm_size > 0:
            pts_norm = pts_norm / palm_size

        coords = pts_norm.flatten()

        # Compute angles between finger joints
        angles = []
        joint_triplets = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),        # thumb
            (0, 5, 6), (5, 6, 7), (6, 7, 8),        # index finger
            (0, 9, 10), (9, 10, 11), (10, 11, 12),  # middle finger
            (0, 13, 14), (13, 14, 15), (15, 16, 16),# ring finger
            (0, 17, 18), (17, 18, 19), (19, 20, 20) # pinky finger
        ]

        for a, b, c in joint_triplets:
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 0 and norm_v2 > 0:
                cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            else:
                angle = 0.0
            angles.append(angle)

        return coords, np.array(angles), pts_norm

    def extract_two_hand_features(self, left_landmarks, right_landmarks):
        """
        Extract combined features from two hands for ISL recognition.
        Normalizes both hands relative to each other for spatial relationship.
        """
        # Extract features for each hand individually
        left_coords, left_angles, left_pts_norm = self.extract_features(left_landmarks)
        right_coords, right_angles, right_pts_norm = self.extract_features(right_landmarks)

        # Compute spatial relationship between hands
        left_wrist = np.array([left_landmarks[0].x, left_landmarks[0].y, left_landmarks[0].z])
        right_wrist = np.array([right_landmarks[0].x, right_landmarks[0].y, right_landmarks[0].z])
        
        # Vector from left wrist to right wrist (normalized by average palm size)
        left_palm = np.linalg.norm(left_pts_norm[9])
        right_palm = np.linalg.norm(right_pts_norm[9])
        avg_palm = (left_palm + right_palm) / 2 if (left_palm + right_palm) > 0 else 1.0
        
        hand_vector = (right_wrist - left_wrist) / avg_palm if avg_palm > 0 else np.zeros(3)
        
        # Relative positions of key landmarks between hands
        # Index fingertips relative to each other
        left_idx_tip = left_pts_norm[8]
        right_idx_tip = right_pts_norm[8]
        tip_vector = right_idx_tip - left_idx_tip
        
        # Combine all features: left_coords(63) + right_coords(63) + left_angles(15) + right_angles(15) + spatial(6) = 162
        combined_coords = np.hstack([left_coords, right_coords])
        combined_angles = np.hstack([left_angles, right_angles])
        spatial_features = np.hstack([hand_vector, tip_vector])
        
        return combined_coords, combined_angles, spatial_features, left_pts_norm, right_pts_norm

    def _is_extended(self, pts_norm, tip_idx, pip_idx, mcp_idx=0):
        """Check if a finger is extended or flexed."""
        dist_tip = np.linalg.norm(pts_norm[tip_idx] - pts_norm[mcp_idx])
        dist_pip = np.linalg.norm(pts_norm[pip_idx] - pts_norm[mcp_idx])
        return dist_tip > dist_pip * 1.15

    def predict_geometric(self, landmarks):
        """
        Geometric rule-based prediction of ASL letters based on hand landmark positions and angles.
        Returns the predicted letter and a confidence score.
        (Single-hand only - used for ASL fallback)
        """
        _, angles, pts = self.extract_features(landmarks)

        # extension status of fingers
        thumb_ext = np.linalg.norm(pts[4] - pts[17]) > np.linalg.norm(pts[2] - pts[17]) * 1.1
        index_ext = self._is_extended(pts, 8, 6, 5)
        middle_ext = self._is_extended(pts, 12, 10, 9)
        ring_ext = self._is_extended(pts, 16, 14, 13)
        pinky_ext = self._is_extended(pts, 20, 18, 17)

        # Key distances between tips
        dist_thumb_index = np.linalg.norm(pts[4] - pts[8])
        dist_thumb_middle = np.linalg.norm(pts[4] - pts[12])
        dist_thumb_ring = np.linalg.norm(pts[4] - pts[16])
        dist_thumb_pinky = np.linalg.norm(pts[4] - pts[20])
        dist_index_middle = np.linalg.norm(pts[8] - pts[12])

        wrist_to_mcp = pts[9] - pts[0]
        pointing_down = wrist_to_mcp[1] > 0.3
        pointing_side = abs(wrist_to_mcp[0]) > abs(wrist_to_mcp[1])

        # Rule for clenched fist: A, E, S, M, N, T
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if pts[4][1] < pts[6][1] or pts[4][0] < pts[5][0]:
                if dist_thumb_index > 0.25 and pts[4][1] > pts[3][1] - 0.2:
                    return "A", 0.92
                elif pts[4][1] > pts[10][1] and dist_thumb_index < 0.3:
                    return "S", 0.90
                elif pts[4][0] > pts[6][0] and pts[4][1] > pts[6][1]:
                    return "E", 0.88
                return "A", 0.85

        # B: All 4 fingers extended, thumb hidden
        if index_ext and middle_ext and ring_ext and pinky_ext:
            if dist_thumb_index < 0.5 and not thumb_ext:
                return "B", 0.94
            return "B", 0.88

        # C: Curved hand in C shape
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if 0.35 < dist_thumb_index < 0.8:
                return "C", 0.89

        # D / L / G / Q
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            if dist_thumb_middle < 0.3 or dist_thumb_ring < 0.3:
                return "D", 0.93
            elif thumb_ext and dist_thumb_index > 0.4:
                if pointing_side:
                    return "G", 0.90
                return "L", 0.95
            elif pointing_down:
                return "Q", 0.88
            return "D", 0.85

        # F: index and thumb touch, other fingers extended
        if not index_ext and middle_ext and ring_ext and pinky_ext:
            if dist_thumb_index < 0.3:
                return "F", 0.94

        # I, Y: pinky extended, others flexed
        if pinky_ext and not index_ext and not middle_ext and not ring_ext:
            if thumb_ext:
                return "Y", 0.95
            return "I", 0.93

        # V, U, R, K, P
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            if dist_index_middle > 0.25:
                if pointing_down:
                    return "P", 0.90
                elif thumb_ext:
                    return "K", 0.91
                return "V", 0.94
            else:
                if pts[8][0] > pts[12][0]:
                    return "R", 0.92
                return "U", 0.93

        # W, M, N
        if index_ext and middle_ext and ring_ext and not pinky_ext:
            if pointing_down:
                return "M", 0.86
            return "W", 0.94

        if thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
            return "Y", 0.95

        return "A", 0.60

    def predict_geometric_two_hand(self, left_landmarks, right_landmarks):
        """
        Geometric rule-based prediction for two-handed ISL signs.
        Implements basic rules for common ISL two-handed letters.
        """
        # Extract features for both hands
        left_coords, left_angles, left_pts = self.extract_features(left_landmarks)
        right_coords, right_angles, right_pts = self.extract_features(right_landmarks)

        # Check finger extensions for both hands
        def get_finger_state(pts):
            return {
                'thumb': np.linalg.norm(pts[4] - pts[17]) > np.linalg.norm(pts[2] - pts[17]) * 1.1,
                'index': np.linalg.norm(pts[8] - pts[5]) > np.linalg.norm(pts[6] - pts[5]) * 1.15,
                'middle': np.linalg.norm(pts[12] - pts[9]) > np.linalg.norm(pts[10] - pts[9]) * 1.15,
                'ring': np.linalg.norm(pts[16] - pts[13]) > np.linalg.norm(pts[14] - pts[13]) * 1.15,
                'pinky': np.linalg.norm(pts[20] - pts[17]) > np.linalg.norm(pts[18] - pts[17]) * 1.15,
            }

        left_fingers = get_finger_state(left_pts)
        right_fingers = get_finger_state(right_pts)

        # Compute spatial relationship
        left_wrist = np.array([left_landmarks[0].x, left_landmarks[0].y, left_landmarks[0].z])
        right_wrist = np.array([right_landmarks[0].x, right_landmarks[0].y, right_landmarks[0].z])
        hand_distance = np.linalg.norm(right_wrist - left_wrist)

        # Relative positions
        left_higher = left_wrist[1] < right_wrist[1]  # y is inverted in image coords
        left_leftof = left_wrist[0] < right_wrist[0]

        # ISL Two-handed letter rules
        # A: Both fists, thumbs out to sides (or one hand A shape, other supporting)
        # This is a simplified heuristic - the ML model will be more accurate
        
        # Both hands fist-like (all fingers folded)
        left_fist = not any([left_fingers['index'], left_fingers['middle'], left_fingers['ring'], left_fingers['pinky']])
        right_fist = not any([right_fingers['index'], right_fingers['middle'], right_fingers['ring'], right_fingers['pinky']])
        
        if left_fist and right_fist:
            if left_fingers['thumb'] and right_fingers['thumb']:
                return "A", 0.75  # Both thumbs extended
        
        # Both hands flat (B shape) - palms facing
        left_flat = all([left_fingers['index'], left_fingers['middle'], left_fingers['ring'], left_fingers['pinky']])
        right_flat = all([right_fingers['index'], right_fingers['middle'], right_fingers['ring'], right_fingers['pinky']])
        
        if left_flat and right_flat:
            if not left_fingers['thumb'] and not right_fingers['thumb']:
                return "B", 0.75

        # Both hands index extended (D/1 shape)
        left_d = left_fingers['index'] and not left_fingers['middle'] and not left_fingers['ring'] and not left_fingers['pinky']
        right_d = right_fingers['index'] and not right_fingers['middle'] and not right_fingers['ring'] and not right_fingers['pinky']
        
        if left_d and right_d:
            return "D", 0.70

        # One hand index up, other hand fist (common pattern)
        if left_d and right_fist:
            return "D", 0.65
        if right_d and left_fist:
            return "D", 0.65

        # V shape both hands
        left_v = left_fingers['index'] and left_fingers['middle'] and not left_fingers['ring'] and not left_fingers['pinky']
        right_v = right_fingers['index'] and right_fingers['middle'] and not right_fingers['ring'] and not right_fingers['pinky']
        
        if left_v and right_v:
            # Check if spread apart (V) or together (U)
            left_tip = left_pts[8]
            right_tip = right_pts[8]
            tip_dist = np.linalg.norm(left_tip - right_tip)
            if tip_dist > 0.3:
                return "V", 0.70
            else:
                return "U", 0.70

        # W shape both hands (3 fingers)
        left_w = left_fingers['index'] and left_fingers['middle'] and left_fingers['ring'] and not left_fingers['pinky']
        right_w = right_fingers['index'] and right_fingers['middle'] and right_fingers['ring'] and not right_fingers['pinky']
        
        if left_w and right_w:
            return "W", 0.70

        # Y shape both hands (thumb + pinky)
        left_y = left_fingers['thumb'] and left_fingers['pinky'] and not left_fingers['index'] and not left_fingers['middle'] and not left_fingers['ring']
        right_y = right_fingers['thumb'] and right_fingers['pinky'] and not right_fingers['index'] and not right_fingers['middle'] and not right_fingers['ring']
        
        if left_y and right_y:
            return "Y", 0.70

        # Index fingers touching (common for many letters)
        left_idx_tip = left_pts[8]
        right_idx_tip = right_pts[8]
        idx_tip_dist = np.linalg.norm(left_idx_tip - right_idx_tip)
        
        if idx_tip_dist < 0.15:
            # Tips touching - could be various letters
            if left_fist and right_fist:
                return "A", 0.60
            if left_d and right_d:
                return "D", 0.60

        # Hands crossed or overlapping (common in ISL)
        if hand_distance < 0.1:
            if left_fist and right_fist:
                return "S", 0.60

        # Default fallback - return empty to let ML model handle it
        return "", 0.0

    def predict(self, landmarks, second_hand_landmarks=None):
        """
        Main prediction function. 
        
        For ASL: Uses single hand (landmarks parameter) -> 78 dim features
        For ISL: 
          - If second_hand_landmarks provided: two-hand features (162 dim)
          - Otherwise: single-hand features padded to 162 dim
        Both use the unified ml_model.
        """
        if self.dataset == "ASL":
            return self._predict_asl_single_hand(landmarks)
        
        # ISL: unified model handles both 1-hand and 2-hand
        if second_hand_landmarks is not None:
            return self._predict_isl_two_hand(landmarks, second_hand_landmarks)
        else:
            return self._predict_isl_single_hand(landmarks)

    def _predict_asl_single_hand(self, landmarks):
        """ASL single-hand prediction (78 dim features)."""
        if not landmarks or len(landmarks) < 21:
            return "", 0.0

        motion_letter, motion_conf = self.motion_recognizer.update(landmarks)
        if motion_letter:
            self.history = [motion_letter]
            return motion_letter, motion_conf

        if self.ml_model is not None:
            try:
                coords, angles, _ = self.extract_features(landmarks)
                feat_vec = np.hstack([coords, angles]).reshape(1, -1)  # 78 dims
                pred = self.ml_model.predict(feat_vec)[0]
                prob = 0.95
                if hasattr(self.ml_model, "predict_proba"):
                    probs = self.ml_model.predict_proba(feat_vec)[0]
                    prob = float(np.max(probs))
                raw_letter, conf = str(pred), prob
            except Exception:
                raw_letter, conf = self.predict_geometric(landmarks)
        else:
            raw_letter, conf = self.predict_geometric(landmarks)

        # Temporal smoothing
        self.history.append(raw_letter)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        from collections import Counter
        counts = Counter(self.history)
        smoothed_letter = counts.most_common(1)[0][0]

        return smoothed_letter, conf

    def _predict_isl_single_hand(self, landmarks):
        """ISL single-hand prediction (78 dim features padded to 162)."""
        if not landmarks or len(landmarks) < 21:
            return "", 0.0

        motion_letter, motion_conf = self.motion_recognizer.update(landmarks)
        if motion_letter:
            self.history = [motion_letter]
            return motion_letter, motion_conf

        if self.ml_model is not None:
            try:
                coords, angles, _ = self.extract_features(landmarks)
                feat_vec = np.hstack([coords, angles])  # 78 dims
                # Pad to 162 dims to match training (two-hand feature size)
                if len(feat_vec) < 162:
                    padding = np.zeros(162 - len(feat_vec))
                    feat_vec = np.hstack([feat_vec, padding])
                feat_vec = feat_vec.reshape(1, -1)
                pred = self.ml_model.predict(feat_vec)[0]
                prob = 0.95
                if hasattr(self.ml_model, "predict_proba"):
                    probs = self.ml_model.predict_proba(feat_vec)[0]
                    prob = float(np.max(probs))
                raw_letter, conf = str(pred), prob
            except Exception as e:
                print(f"[SignRecognizer] ISL single-hand model error: {e}")
                raw_letter, conf = self.predict_geometric(landmarks)
        else:
            raw_letter, conf = self.predict_geometric(landmarks)

        # Temporal smoothing
        self.history.append(raw_letter)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        from collections import Counter
        counts = Counter(self.history)
        smoothed_letter = counts.most_common(1)[0][0]

        return smoothed_letter, conf

    def _predict_isl_two_hand(self, left_landmarks, right_landmarks):
        """ISL two-hand prediction (162 dim features)."""
        if not left_landmarks or not right_landmarks or len(left_landmarks) < 21 or len(right_landmarks) < 21:
            return "", 0.0

        # Use unified ml_model with two-hand features
        if self.ml_model is not None:
            try:
                coords, angles, spatial, _, _ = self.extract_two_hand_features(left_landmarks, right_landmarks)
                feat_vec = np.hstack([coords, angles, spatial]).reshape(1, -1)  # 162 dims
                pred = self.ml_model.predict(feat_vec)[0]
                prob = 0.95
                if hasattr(self.ml_model, "predict_proba"):
                    probs = self.ml_model.predict_proba(feat_vec)[0]
                    prob = float(np.max(probs))
                raw_letter, conf = str(pred), prob
            except Exception as e:
                print(f"[SignRecognizer] ISL two-hand model error: {e}")
                raw_letter, conf = self.predict_geometric_two_hand(left_landmarks, right_landmarks)
        else:
            raw_letter, conf = self.predict_geometric_two_hand(left_landmarks, right_landmarks)

        # Temporal smoothing
        self.history.append(raw_letter)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        from collections import Counter
        counts = Counter(self.history)
        smoothed_letter = counts.most_common(1)[0][0]

        return smoothed_letter, conf