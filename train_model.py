import os
import cv2
import numpy as np
import joblib
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ['NORMAL', 'FILAMENT_NOT_DEPOSITED', 'FILAMENT_NOT_ADHERING', 'UNCERTAIN']

def load_image_safe(path):
    """Unicode-safe image reader for Windows file paths."""
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path)

def get_augmented_images(img, category):
    """Generate physics-aware augmented images tailored to 3D printing failure modes."""
    images = [img]
    
    (h, w) = img.shape[:2]
    
    if category in ['normal', 'correct']:
        # Simulate slight lighting changes & scale
        darker = cv2.convertScaleAbs(img, alpha=0.85, beta=-15)
        brighter = cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        images.extend([darker, brighter])
        
    elif category in ['filament_not_deposited', 'under']:
        # Simulate severe line erosion & gap insertion
        kernel = np.ones((3, 3), np.uint8)
        eroded1 = cv2.erode(img, kernel, iterations=1)
        eroded2 = cv2.erode(img, kernel, iterations=2)
        images.extend([eroded1, eroded2])
        
    elif category in ['filament_not_adhering', 'over', '0ver']:
        # Simulate thread stretching / warping (sine wave shift simulating floating thread)
        rows, cols = img.shape[:2]
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(5.0 * np.sin(2 * 3.14159 * i / 60))
                if j + offset_x < cols and j + offset_x >= 0:
                    img_output[i, j] = img[i, (j + offset_x) % cols]
                else:
                    img_output[i, j] = 0
        images.append(img_output)
        
    return images

def extract_features_from_img(img):
    """Extracts 8 physical features targeted at extrusion & bed adhesion detection."""
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    edges = cv2.Canny(thresh, 50, 150)
    
    total_pixels = 200 * 200
    edge_ratio = np.sum(edges > 0) / float(total_pixels)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour_area = 0.0
    thread_wavy_index = 0.0
    
    if contours:
        contour_areas = [cv2.contourArea(c) for c in contours]
        max_contour_area = max(contour_areas)
        largest_c = contours[np.argmax(contour_areas)]
        
        # Measure contour aspect ratio and perimeter complexity for thread detection
        perimeter = cv2.arcLength(largest_c, True)
        if max_contour_area > 0:
            thread_wavy_index = (perimeter * perimeter) / (4 * np.pi * max_contour_area)
            
    edge_continuity = 0.0
    if np.sum(edges > 0) > 0:
        edge_continuity = 1.0 - min(1.0, len(contours) / (np.sum(edges > 0) / 255.0))
        
    # Feature 7: Nozzle tip orifice region intensity (top-center ROI 80x40)
    nozzle_roi = gray[40:80, 80:120]
    nozzle_orifice_intensity = np.mean(nozzle_roi) if nozzle_roi.size > 0 else 0.0
    
    # Feature 8: Edge density along expected vertical print path (x: 90..110, y: 80..180)
    path_roi = edges[80:180, 90:110]
    expected_path_density = np.sum(path_roi > 0) / float(path_roi.size) if path_roi.size > 0 else 0.0
    
    return [
        edge_ratio,
        mean_intensity,
        std_intensity,
        max_contour_area,
        edge_continuity,
        thread_wavy_index,
        nozzle_orifice_intensity,
        expected_path_density
    ]

def train_rf(dataset_path, labels, save_path):
    print("Extracting features with physics-aware data augmentations for Random Forest...")
    X, y = [], []
    
    for category, class_id in labels.items():
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue
            
        print(f" -> Processing '{category}' (Class {class_id})...")
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = load_image_safe(path)
            if img is not None:
                aug_images = get_augmented_images(img, category)
                for aug_img in aug_images:
                    feats = extract_features_from_img(aug_img)
                    if feats is not None:
                        X.append(feats)
                        y.append(class_id)

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("[ERROR] No valid images found for training.")
        return

    print(f"Total training dataset size: {len(X)} samples across {len(np.unique(y))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"\n=========================================")
    print(f" Random Forest Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"=========================================")
    
    target_names = [CLASS_NAMES[i] for i in sorted(np.unique(y))]
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Model successfully saved to {save_path}")

def train_cnn(dataset_path, labels, save_path):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        print("[WARN] TensorFlow/Keras not installed. Run `pip install tensorflow` to use CNN model.")
        return

    print("Loading image data for CNN model...")
    X, y = [], []
    num_classes = len(set(labels.values()))
    
    for category, class_id in labels.items():
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue
            
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = load_image_safe(path)
            if img is not None:
                img_resized = cv2.resize(img, (128, 128))
                img_norm = img_resized.astype('float32') / 255.0
                X.append(img_norm)
                y.append(class_id)

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No valid images found for training.")
        return

    y_cat = to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), batch_size=16)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Accuracy: {accuracy:.4f}")

    cnn_save_path = save_path.replace('.pkl', '.h5')
    model.save(cnn_save_path)
    print(f"CNN Model saved as {cnn_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Printer First Layer Defect Model")
    parser.add_argument("--model", type=str, choices=["rf", "cnn"], default="rf", 
                        help="Choose 'rf' for Random Forest or 'cnn' for TensorFlow CNN.")
    args = parser.parse_args()

    dataset_path = "dataset"
    save_path = os.path.join("models", "model.pkl")

    labels = {
        "normal": 0,
        "correct": 0,
        "filament_not_deposited": 1,
        "under": 1,
        "filament_not_adhering": 2,
        "over": 2,
        "0ver": 2,
        "uncertain": 3
    }

    if args.model == "rf":
        train_rf(dataset_path, labels, save_path)
    elif args.model == "cnn":
        train_cnn(dataset_path, labels, save_path)