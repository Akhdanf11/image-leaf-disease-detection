from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('mango_leaf_disease_model.h5')

# Class labels
class_labels = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
    'Die Back', 'Gall Midge', 'Healthy', 
    'Powdery Mildew', 'Sooty Mould'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values
        
        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        
        # Confidence score for each class
        confidence_scores = {class_labels[i]: f"{predictions[0][i] * 100:.2f}%" for i in range(len(class_labels))}
        
        return jsonify({
            'filename': filename,
            'predicted_class': predicted_class_label,
            'confidence_scores': confidence_scores
        })

if __name__ == '__main__':
    app.run(debug=True)
