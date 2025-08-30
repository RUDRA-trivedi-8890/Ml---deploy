import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models once at startup using pickle
with open("svm_mangrove_detector.pkl", "rb") as f:
    svm_clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

cnn_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x)
    return features.flatten().reshape(1, -1)

@app.route('/', methods=['GET'])
def home():
    return jsonify("Hello! You are at the home page.")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # Extract features and predict
        features = extract_features(img_path)
        pred_idx = svm_clf.predict(features)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        pred_proba = svm_clf.predict_proba(features).max()

        # Remove image after prediction
        os.remove(img_path)

        return jsonify({
            'prediction': pred_label,
            'confidence': float(pred_proba)
        })

    else:
        return jsonify({'error': 'Unsupported file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
