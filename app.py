from flask import Flask, request, jsonify, render
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore
import io
from PIL import Image # type: ignore

app = Flask(__name__)

# Load the trained model
model = load_model('cats_dogs_classifier.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img = file.read()
    img_array = prepare_image(img)
    
    prediction = model.predict(img_array)
    class_label = 'Cat' if prediction[0] > 0.5 else 'Dog'
    
    return jsonify({'prediction': class_label})

if __name__ == '__main__':
    app.run(debug=True)
