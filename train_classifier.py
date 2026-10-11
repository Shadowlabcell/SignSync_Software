"""
Script for training machine learning models for ASL/ISL sign recognition.

There are three independent pipelines here:

  1. Static pose pipeline (option 1/2): collects hand-landmark snapshots
     per sample for the letters that are a fixed handshape, and trains
     the Random Forest used by SignLanguageRecognizer.
     - ASL: Single-hand only (num_hands=1)
     - ISL: Flexible - accepts 1 or 2 hands (num_hands=2), extracts
       appropriate features based on hands detected

  2. Motion pipeline (option 3/4): collects a short *sequence* of frames per
     sample for letters that are drawn as a movement (not a pose), and
     trains the small Random Forest used by MotionGestureRecognizer.

Dataset-specific motion letters:
- ASL: J, Z (drawn as movements in the air)
- ISL: H, J (drawn as movements in Indian Sign Language)

Static letters are all remaining A-Z letters not in the motion set for that dataset.
"""

import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app.sign_recognizer import SignLanguageRecognizer
from app.motion_recognizer import (
    BUFFER_SIZE as MOTION_BUFFER_SIZE,
    palm_size as motion_palm_size,
    extract_trajectory_features,
    FINGERTIP_INDEX,
)
from app.config import (
    DATASETS,
    get_model_paths,
    set_current_dataset,
    ensure_dataset_dirs,
    get_current_dataset,
)

# Dataset-specific motion letters (letters drawn as movements, not static poses)
MOTION_LETTERS_BY_DATASET = {
    "ASL": ["J", "Z"],      # ASL: J and Z are motion-based
    "ISL": ["H", "J"],      # ISL: H and J are motion-based
}

def get_motion_letters(dataset: str) -> list:
    """Get motion letters for a specific dataset."""
    return MOTION_LETTERS_BY_DATASET.get(dataset, ["J", "Z"])

def get_static_letters(dataset: str) -> list:
    """Get static letters for a specific dataset (all A-Z except motion letters)."""
    motion = get_motion_letters(dataset)
    return [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in motion]


def _ensure_hand_model():
    os.makedirs("task", exist_ok=True)
    hand_model_path = os.path.join("task", "hand_landmarker.task")
    if not os.path.exists(hand_model_path):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_model_path)
    return hand_model_path


def collect_dataset(dataset: str, output_file=None, samples_per_class=150):
    """
    Collects static, single-frame samples for every letter EXCEPT motion letters.
    
    For ASL: Uses num_hands=1, extracts single-hand features (78 dims)
    For ISL: Uses num_hands=2, flexibly handles 1 or 2 hands:
      - If 2 hands detected: extracts two-hand features (162 dims)
      - If 1 hand detected: extracts single-hand features + zero-pads to 162 dims
    This allows the ISL model to recognize both 1-hand and 2-hand signs.
    """
    motion_letters = get_motion_letters(dataset)
    static_letters = get_static_letters(dataset)

    if output_file is None:
        paths = get_model_paths(dataset)
        output_file = paths["sign_dataset"]

    print(f"=== Dataset Collection for {dataset} (static letters) ===")
    print(f"Static letters: {', '.join(static_letters)}")
    print(f"Motion letters (collected separately): {', '.join(motion_letters)}")
    print(f"Samples per class: {samples_per_class}")

    pkl_dir = os.path.dirname(output_file)
    os.makedirs(pkl_dir, exist_ok=True)

    # For ISL, start fresh to avoid dimension mismatches from old data
    existing_data = []
    existing_labels = []
    if os.path.exists(output_file):
        print(f"Existing dataset found at '{output_file}'. Starting fresh to ensure consistent feature dimensions.")
        # Optionally, you could load and filter, but for ISL with new 162-dim features,
        # it's safer to start fresh. For ASL, we can append.
        if dataset == "ASL":
            with open(output_file, "rb") as f:
                existing = pickle.load(f)
            raw_data = list(existing["data"])
            raw_labels = list(existing["labels"])

            dropped = 0
            for d, l in zip(raw_data, raw_labels):
                if l in motion_letters:
                    dropped += 1
                    continue
                existing_data.append(d)
                existing_labels.append(l)

            if dropped:
                print(f"Removed {dropped} old motion letter sample(s).")
            print(f"Loaded {len(existing_labels)} existing ASL samples.")
        else:
            print("ISL dataset will be recreated with new 162-dim features.")

    hand_model_path = _ensure_hand_model()
    
    # For ISL, use 2 hands to capture both 1-hand and 2-hand signs
    # For ASL, use 1 hand
    num_hands = 2 if dataset == "ISL" else 1
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=num_hands
    )
    detector = vision.HandLandmarker.create_from_options(options)
    recognizer = SignLanguageRecognizer()

    cap = cv2.VideoCapture(0)
    data = existing_data
    labels = existing_labels

    # Feature dimension: 78 for single-hand (ASL), 162 for ISL (two-hand + padded single-hand)
    TARGET_DIM = 162 if dataset == "ISL" else 78

    for letter in static_letters:
        print(f"\nPrepare for letter: {letter}. Press 'SPACE' to start recording...")
        if dataset == "ISL":
            print("  (Show the sign naturally - 1 hand or 2 hands both work)")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hint = " (1 or 2 hands)" if dataset == "ISL" else ""
            cv2.putText(frame, f"Letter: {letter} - Press SPACE to start recording{hint}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if dataset == "ISL":
                cv2.putText(frame, "Show sign naturally - system detects hand count automatically", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Dataset Collector", frame)
            key = cv2.waitKey(1)
            if key == 32:  # Space
                break
            elif key == 27:  # Esc
                cap.release()
                cv2.destroyAllWindows()
                if len(labels) > len(existing_labels):
                    with open(output_file, "wb") as f:
                        pickle.dump({"data": np.array(data), "labels": np.array(labels)}, f)
                    print(f"Stopped early. Progress saved to {output_file} "
                          f"({len(labels)} total samples).")
                else:
                    print("Stopped early. No new samples were collected, nothing changed.")
                return

        collected = 0
        hand_mode = "1H"  # Default
        while collected < samples_per_class:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

            feat_vec = None
            if res.hand_landmarks:
                # For ISL: use all detected hands (1 or 2)
                # For ASL: use first hand only
                if dataset == "ISL" and len(res.hand_landmarks) >= 2:
                    # Two hands detected - extract two-hand features
                    left_hand = res.hand_landmarks[0]
                    right_hand = res.hand_landmarks[1]
                    left_wrist_x = left_hand[0].x
                    right_wrist_x = right_hand[0].x
                    if left_wrist_x > right_wrist_x:
                        left_hand, right_hand = right_hand, left_hand
                    
                    coords, angles, spatial, _, _ = recognizer.extract_two_hand_features(left_hand, right_hand)
                    feat_vec = np.hstack([coords, angles, spatial])  # 162 dims
                    hand_mode = "2H"
                else:
                    # Single hand (ASL or ISL with 1 hand)
                    hand_landmarks = res.hand_landmarks[0]
                    coords, angles, _ = recognizer.extract_features(hand_landmarks)
                    feat_vec = np.hstack([coords, angles])  # 78 dims
                    
                    # For ISL, pad to 162 dims to match two-hand feature size
                    if dataset == "ISL" and len(feat_vec) < 162:
                        padding = np.zeros(162 - len(feat_vec))
                        feat_vec = np.hstack([feat_vec, padding])
                    hand_mode = "1H"

                # Verify feature dimension
                if feat_vec is not None and len(feat_vec) != TARGET_DIM:
                    print(f"WARNING: Feature dimension mismatch! Got {len(feat_vec)}, expected {TARGET_DIM}. Skipping sample.")
                    feat_vec = None

            if feat_vec is not None:
                data.append(feat_vec)
                labels.append(letter)
                collected += 1

            # Display progress
            cv2.putText(frame, f"Collecting {letter} [{hand_mode}]: {collected}/{samples_per_class}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            if dataset == "ISL":
                cv2.putText(frame, f"Detected hands: {len(res.hand_landmarks) if res.hand_landmarks else 0}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Dataset Collector", frame)
            cv2.waitKey(20)

    cap.release()
    cv2.destroyAllWindows()

    # Final verification before saving
    if len(data) > 0:
        data_array = np.array(data)
        print(f"Final dataset shape: {data_array.shape}")
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.ndim}D"
        assert data_array.shape[1] == TARGET_DIM, f"Expected {TARGET_DIM} features, got {data_array.shape[1]}"
    
    with open(output_file, "wb") as f:
        pickle.dump({"data": np.array(data), "labels": np.array(labels)}, f)
    print(f"\nDataset saved successfully to {output_file}! ({len(labels)} total samples, {TARGET_DIM} features)")


def collect_motion_dataset(dataset: str, output_file=None, samples_per_class=60):
    """
    Collects motion samples for motion letters. Each sample is a short sequence of
    MOTION_BUFFER_SIZE consecutive frames recorded right after SPACE is
    pressed, capturing the whole stroke rather than a single pose.
    """
    motion_letters = get_motion_letters(dataset)

    if output_file is None:
        paths = get_model_paths(dataset)
        output_file = paths["motion_dataset"]

    print(f"=== Dataset Collection for {dataset} (motion letters: {', '.join(motion_letters)}) ===")
    print(f"Each sample records {MOTION_BUFFER_SIZE} frames - perform the full "
          f"gesture right after pressing SPACE and keep going until it stops recording.")
    print(f"Samples per class: {samples_per_class}")

    pkl_dir = os.path.dirname(output_file)
    os.makedirs(pkl_dir, exist_ok=True)

    existing_data = []
    existing_labels = []
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            existing = pickle.load(f)
        existing_data = list(existing["data"])
        existing_labels = list(existing["labels"])
        print(f"Found existing motion dataset '{output_file}' with {len(existing_labels)} samples. "
              f"New samples will be appended to it.")

    hand_model_path = _ensure_hand_model()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    data = existing_data
    labels = existing_labels

    for letter in motion_letters:
        print(f"\nPrepare for letter: {letter}. Press 'SPACE' to start recording a stroke...")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Letter: {letter} - Press SPACE to record a full gesture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Motion Dataset Collector", frame)
            key = cv2.waitKey(1)
            if key == 32:  # Space
                break
            elif key == 27:  # Esc
                cap.release()
                cv2.destroyAllWindows()
                if len(labels) > len(existing_labels):
                    with open(output_file, "wb") as f:
                        pickle.dump({"data": np.array(data), "labels": np.array(labels)}, f)
                    print(f"Stopped early. Progress saved to {output_file} "
                          f"({len(labels)} total samples).")
                else:
                    print("Stopped early. No new samples were collected, nothing changed.")
                return

        collected = 0
        while collected < samples_per_class:
            stroke_points = []
            frames_needed = MOTION_BUFFER_SIZE
            while frames_needed > 0:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_img)

                if res.hand_landmarks:
                    lms = res.hand_landmarks[0]
                    palm = motion_palm_size(lms)
                    tip = lms[FINGERTIP_INDEX]
                    stroke_points.append((tip.x / palm, tip.y / palm))
                    frames_needed -= 1

                cv2.putText(frame, f"Recording {letter}: sample {collected + 1}/{samples_per_class}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                cv2.imshow("Motion Dataset Collector", frame)
                cv2.waitKey(20)

            if len(stroke_points) == MOTION_BUFFER_SIZE:
                features = extract_trajectory_features(np.array(stroke_points))
                data.append(features)
                labels.append(letter)
                collected += 1
            # if the hand was lost mid-stroke, the sample is simply discarded
            # and this iteration retries the same letter/count

    cap.release()
    cv2.destroyAllWindows()

    with open(output_file, "wb") as f:
        pickle.dump({"data": np.array(data), "labels": np.array(labels)}, f)
    print(f"\nMotion dataset saved successfully to {output_file}! ({len(labels)} total samples)")


def train_model(dataset: str, dataset_file=None, model_file=None):
    paths = get_model_paths(dataset)
    if dataset_file is None:
        dataset_file = paths["sign_dataset"]
    if model_file is None:
        model_file = paths["sign_model"]

    if not os.path.exists(dataset_file):
        print(f"File {dataset_file} does not exist! Please collect the data first.")
        return

    pkl_dir = os.path.dirname(model_file)
    os.makedirs(pkl_dir, exist_ok=True)

    with open(dataset_file, "rb") as f:
        dataset_data = pickle.load(f)

    X = list(dataset_data["data"])
    y = list(dataset_data["labels"])

    # Defensive filter: never let motion letters train the static model, even if an
    # older dataset file still has them in it.
    motion_letters = get_motion_letters(dataset)
    filtered = [(xi, yi) for xi, yi in zip(X, y) if yi not in motion_letters]
    if len(filtered) < len(X):
        print(f"Ignoring {len(X) - len(filtered)} motion letter sample(s) found in the static dataset.")
    X, y = [xi for xi, _ in filtered], [yi for _, yi in filtered]

    if len(X) == 0:
        print("No training samples available after filtering!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use more estimators for ISL since it has more complex (1-hand + 2-hand) patterns
    n_estimators = 200 if dataset == "ISL" else 150
    max_depth = 15 if dataset == "ISL" else 12

    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Trained {dataset} Static Model Accuracy: {acc * 100:.2f}%")
    print(f"  Classes: {len(np.unique(y))}, Samples: {len(X)}, Features: {X[0].shape if len(X) > 0 else 'N/A'}")

    with open(model_file, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved successfully to {model_file}!")


def train_motion_model(dataset: str, dataset_file=None, model_file=None):
    paths = get_model_paths(dataset)
    if dataset_file is None:
        dataset_file = paths["motion_dataset"]
    if model_file is None:
        model_file = paths["motion_model"]

    if not os.path.exists(dataset_file):
        print(f"File {dataset_file} does not exist! Please collect motion data first (option 3).")
        return

    pkl_dir = os.path.dirname(model_file)
    os.makedirs(pkl_dir, exist_ok=True)

    with open(dataset_file, "rb") as f:
        dataset_data = pickle.load(f)

    X = dataset_data["data"]
    y = dataset_data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    motion_letters = get_motion_letters(dataset)
    print(f"Trained {dataset} Motion Model Accuracy ({', '.join(motion_letters)}): {acc * 100:.2f}%")

    with open(model_file, "wb") as f:
        pickle.dump(clf, f)
    print(f"Motion model saved successfully to {model_file}!")


def select_dataset():
    """Prompt user to select dataset (ASL/ISL)."""
    print("\nSelect Sign Language Dataset:")
    for i, ds in enumerate(DATASETS, 1):
        motion = get_motion_letters(ds)
        static = get_static_letters(ds)
        print(f"  {i}. {ds} (Static: {len(static)} letters, Motion: {', '.join(motion)})")
    while True:
        choice = input("Choose dataset (1/2): ").strip()
        if choice == "1":
            return "ASL"
        elif choice == "2":
            return "ISL"
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    ensure_dataset_dirs()

    # First, ask for dataset
    dataset = select_dataset()
    set_current_dataset(dataset)
    print(f"\nSelected dataset: {dataset}")

    motion_letters = get_motion_letters(dataset)
    static_letters = get_static_letters(dataset)

    # Unified 4 options for both ASL and ISL
    print(f"\n1. Collect New Static Dataset ({', '.join(static_letters)})")
    print("2. Train Static Model from Existing Dataset")
    print(f"3. Collect New Motion Dataset ({', '.join(motion_letters)} only)")
    print("4. Train Motion Model from Existing Motion Dataset")
    choice = input("Choose option (1/2/3/4): ").strip()
    
    if choice == "1":
        collect_dataset(dataset)
        train_model(dataset)
        print("Done! The static model has been saved and will be loaded automatically by the application.")
    elif choice == "2":
        train_model(dataset)
    elif choice == "3":
        collect_motion_dataset(dataset)
        train_motion_model(dataset)
        print("Done! The motion model has been saved and will be loaded automatically by the application.")
    elif choice == "4":
        train_motion_model(dataset)
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, or 4.")