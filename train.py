import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Paths to dataset folders
dataset_path = r'C:\Users\asus\codes\temp\dataset'
categories = ["mangrove", "nonmangrove"]

# Load pre-trained CNN (MobileNetV2 without top layers)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

# Collect features and labels
features = []
labels = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            feat = extract_features(filepath)
            features.append(feat)
            labels.append(category)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

features = np.array(features)
labels = np.array(labels)

# Encode labels to numeric
le = LabelEncoder()
y = le.fit_transform(labels)

# Train SVM with linear kernel
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(features, y)

# Save model and label encoder for deployment using pickle
with open("svm_mangrove_detector.pkl", "wb") as f:
    pickle.dump(svm_clf, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Training complete and model saved.")
