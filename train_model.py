import os
import cv2
import numpy as np
import joblib
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_augmented_images(img, category):
    """Generate physics-aware augmented images depending on defect category."""
    images = [img]
    
    if category == 'correct':
        # Simulate rotation
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        for angle in [90, 180, 270]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            images.append(rotated)
        # Simulate lighting
        darker = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
        brighter = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        images.extend([darker, brighter])
        
    elif category == 'under':
        # Simulate gaps / broken lines using Morphological Erosion
        kernel1 = np.ones((3,3), np.uint8)
        eroded1 = cv2.erode(img, kernel1, iterations=1)
        eroded2 = cv2.erode(img, kernel1, iterations=2)
        images.extend([eroded1, eroded2])
        
    elif category in ['0ver', 'over']:
        # Simulate thick lines / blobs using Morphological Dilation
        kernel = np.ones((3,3), np.uint8)
        dilated1 = cv2.dilate(img, kernel, iterations=1)
        dilated2 = cv2.dilate(img, kernel, iterations=2)
        images.extend([dilated1, dilated2])
        
    return images

def extract_features_from_img(img):
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Same improved features as backend/detector.py
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    edges = cv2.Canny(thresh, 50, 150)
    
    edge_ratio = np.sum(edges > 0) / (200 * 200)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_contour_area = 0
    if contours:
        max_contour_area = max([cv2.contourArea(c) for c in contours])
        
    edge_continuity = 0.0
    if np.sum(edges > 0) > 0:
        edge_continuity = 1.0 - min(1.0, len(contours) / (np.sum(edges > 0) / 255.0))
        
    return [edge_ratio, mean_intensity, std_intensity, max_contour_area, edge_continuity]

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_features_from_img(img)

def extract_cnn_image_from_img(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    return img

def extract_cnn_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    return extract_cnn_image_from_img(img)

def train_rf(dataset_path, labels, save_path):
    print("Extracting features (with Physics-Aware Data Augmentation) for Random Forest...")
    X, y = [], []
    
    for category in labels:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue
            
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = cv2.imread(path)
            if img is not None:
                # Dynamically augment dataset in-memory based on physics rules
                augmented_images = get_augmented_images(img, category)
                for aug_img in augmented_images:
                    features = extract_features_from_img(aug_img)
                    if features is not None:
                        X.append(features)
                        y.append(labels[category])

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No valid images found for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, pred))
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    # Ensure model dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved as {save_path}")

def train_cnn(dataset_path, labels, save_path):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        print("Tensorflow/Keras not installed. Run `pip install tensorflow` to use CNN model.")
        return
        
    print("Loading image data for CNN...")
    X, y = [], []
    
    for category in labels:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            continue
            
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = cv2.imread(path)
            if img is not None:
                augmented_images = get_augmented_images(img, category)
                for aug_img in augmented_images:
                    cnn_img = extract_cnn_image_from_img(aug_img)
                    if cnn_img is not None:
                        X.append(cnn_img)
                        y.append(labels[category])

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("No valid images found for training.")
        return

    # Convert y to one-hot for keras
    y_cat = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    print("Building simple categorical CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Accuracy: {accuracy:.4f}")
    print(f"CNN Loss: {loss:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nCNN Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    print("CNN Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Switch save path extension for CNN model
    cnn_save_path = save_path.replace('.pkl', '.h5')
    model.save(cnn_save_path)
    print(f"CNN Model saved as {cnn_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Defect Detection Model")
    parser.add_argument("--model", type=str, choices=["rf", "cnn"], default="rf", 
                        help="Choose 'rf' for Random Forest or 'cnn' for TensorFlow CNN.")
    args = parser.parse_args()

    dataset_path = "dataset"
    save_path = os.path.join("models", "model.pkl")

    labels = {
        "correct": 0,
        "under": 1,
        "0ver": 2,      # Existing dataset folder typo supported
        "over": 2       # Adding normalized label mapping as well
    }

    if args.model == "rf":
        train_rf(dataset_path, labels, save_path)
    elif args.model == "cnn":
        train_cnn(dataset_path, labels, save_path)