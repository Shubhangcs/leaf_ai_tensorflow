from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import io

# ✅ Import the legacy model loader for .h5 files
from keras.saving.legacy.hdf5_format import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ✅ Load model using legacy loader
model = load_model('plant_identification_model2.h5')

# Label mapping
label_mapping = {
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Image preprocessing
def preprocess_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        preprocessed_image = preprocess_image_bytes(image_bytes)
        
        print("Image preprocessed, making prediction...")
        predictions = model.predict(preprocessed_image)
        print(f"Predictions: {predictions}")
        
        predicted_index = np.argmax(predictions)
        predicted_label = label_mapping.get(predicted_index, "Unknown")
        confidence = float(predictions[0][predicted_index])
        
        print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")
        
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
