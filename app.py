from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

app = Flask(__name__)

# Define paths and settings
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'weights'

# Define the upload directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the models
model_elbow_frac = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "ResNet50_Elbow_frac.h5"))
model_hand_frac = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "ResNet50_Hand_frac.h5"))
model_shoulder_frac = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "ResNet50_Shoulder_frac.h5"))
model_parts = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "ResNet50_BodyParts.h5"))

# Categories for each result by index
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['Fractured', 'Not Fractured']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Implement logic for processing the uploaded file
        prediction_result = predict(filename)
        return render_template('result.html', prediction_result=prediction_result)

def predict(img_path):
    size = 224
    # Load the image
    temp_img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    # Predict bone type
    bone_type_prediction = categories_parts[np.argmax(model_parts.predict(images), axis=1)[0]]
    
    # Predict fracture status for each part
    if bone_type_prediction == 'Elbow':
        chosen_model = model_elbow_frac
    elif bone_type_prediction == 'Hand':
        chosen_model = model_hand_frac
    elif bone_type_prediction == 'Shoulder':
        chosen_model = model_shoulder_frac
    else:
        return "Error: Invalid bone type prediction"

    fracture_prediction = categories_fracture[np.argmax(chosen_model.predict(images), axis=1)[0]]
    
    return {'bone_type': bone_type_prediction, 'fracture_status': fracture_prediction}

if __name__ == '__main__':
    app.run(debug=True)
