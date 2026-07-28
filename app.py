from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil
import requests

app = Flask(__name__)

# Enable CORS for frontend on Vercel to communicate with this backend
CORS(app, origins=["https://plantpulseaigpk.vercel.app", "https://*.vercel.app", "http://localhost:3000", "http://localhost:5000"])

# Load the pre-trained model
model_path = 'cotton_disease_model.h5'  # Update this with your actual model file path
model = tf.keras.models.load_model(model_path)

# Class names based on the dataset (include 'Unknown' for unrelated images)
class_names = ['Aphids', 'Army worm', 'Bacterial blight', 
               'Healthy leaf', 'Powder mildew', 'Target spot', 'Unknown']

# Solutions dictionary
solutions = {
    "Aphids": {
        "Symptoms": [
            "Tiny green or black insects on the underside of leaves",
            "Leaves curl, turn yellow, and become weak",
            "Sticky honeydew on leaves, leading to black mold"
        ],
        "reason": [
            "Hot and dry weather encourages aphid growth",
            "Excess nitrogen fertilizer attracts aphids",
            "Lack of natural predators like ladybugs"
        ],
        "Effects": [
            "Reduces plant strength and slows growth",
            "Honeydew leads to black mold, blocking sunlight",
            "Lowers cotton yield and fiber quality"
        ],
        "Treatment": {
            "Organic": [
                "Spray Neem oil (5 ml per liter of water)",
                "Introduce ladybugs to the field",
                "Use soap-water spray (1 spoon of dish soap per liter of water)"
            ],
            "Chemical": [
                "Imidacloprid 17.8% SL (3 ml per 10 liters of water)",
                "Thiamethoxam 25% WG (1 g per liter of water)"
            ]
        },
        "Products": [
            {'name': 'Neem Oil Spray', 'url': 'https://www.amazon.in/gp/aw/d/B09ZRP5VMM/', 'image_url': '/static/images/neem-oil.jpg'},
            {'name': 'Imidacloprid 17.8% SL', 'url': 'https://www.amazon.in/s?k=Ulala+upl+comp', 'image_url': '/static/images/ulala.jpg'}
        ],
        "Precautions": [
            "Wear gloves, mask, and goggles",
            "Spray in the early morning or evening",
            "Avoid eating or drinking while spraying"
        ]
    },
    "Army worm": {
        "Symptoms": [
            "Leaves appear severely damaged and ragged",
            "Presence of larvae feeding on leaves"
        ],
        "reason": [
            "Infestation increases in warm, dry seasons",
            "Overuse of fertilizers attracts armyworms"
        ],
        "Effects": [
            "Severe leaf damage reduces photosynthesis",
            "Leads to stunted plant growth and lower yield"
        ],
        "Treatment": {
            "Organic": [
                "Apply Bacillus thuringiensis (Bt)",
                "Release Trichogramma wasps to control larvae"
            ],
            "Chemical": [
                "Spray Chlorantraniliprole 18.5% SC",
                "Spray Lambda-cyhalothrin"
            ]
        },
        "Products": [
            {'name': 'Bt-based Spray', 'url': 'https://www.amazon.in/PLANTS-BUDDY-Thuringiensis-Lepidopteran-Bio-Insecticide/dp/B0CQQSW51X', 'image_url': '/static/images/bacillus.jpg'},
            {'name': 'Chlorantraniliprole 18.5% SC', 'url': 'https://www.amazon.in/SWASTIK-ICON-Insecticide-Lambda-Cyhalothrin/dp/B07ZTJM7VR', 'image_url': '/static/images/coragen.jpg'}
        ],
        "Precautions": [
            "Wear protective clothing and a mask",
            "Avoid spraying during pollination times",
            "Prevent pesticide runoff into water sources"
        ]
    },
    "Bacterial blight": {  # Fixed missing colon
        "Symptoms": [
            "Small water-soaked spots on leaves turning reddish-brown",
            "Veins turn black, and leaves dry out",
            "Bolls develop black spots and fall off"
        ],
        "reason": [
            "Bacteria spread through infected seeds, rain, and insects",
            "Poor drainage and overcrowding increase risk"
        ],
        "Effects": [
            "Premature boll drop reduces yield",
            "Weakens plants, making them prone to other diseases"
        ],
        "Treatment": {
            "Organic": [
                "Use disease-free seeds",
                "Apply Garlic Extract Spray (crushed garlic in water)",
                "Remove and burn infected plants"
            ],
            "Chemical": [
                "Copper Oxychloride (3 g per liter of water)",
                "Streptomycin sulfate (100 g) + Copper Oxychloride (500 g per acre) for severe cases"
            ]
        },
        "Products": [
            {'name': 'Copper Fungicide', 'url': 'https://www.desertcart.in/Products/2343720-bonidecaptain-jack-s-copper-fungicide', 'image_url': '/static/images/liquid-copper-concentrate.jpg'},
            {'name': 'Streptomycin Sulfate Powder', 'url': 'https://www.amazon.in/s?k=streptocycline+vector', 'image_url': '/static/images/vector.jpg'}
        ],
        "Precautions": [
            "Wear protective clothing and a mask",
            "Wash hands after handling chemicals",
            "Store pesticides away from food"
        ]
    },
    "Target spot": {
        "Symptoms": [
            "Brown circular spots on leaves with dark edges",
            "Leaves turn yellow and drop early",
            "Stunted plant growth"
        ],
        "reason": [
            "Fungal spores spread via wind and rain",
            "High humidity and overwatering worsen the problem"
        ],
        "Effects": [
            "Leaf loss reduces photosynthesis",
            "Poor cotton development"
        ],
        "Treatment": {
            "Organic": [
                "Spray Baking Soda Solution (1 tsp baking soda + 1 liter of water)",
                "Remove infected leaves",
                "Ensure proper plant spacing for airflow"
            ],
            "Chemical": [
                "Mancozeb 75% WP (2 g per liter of water)",
                "Carbendazim 50% WP (1 g per liter of water)"
            ]
        },
        "Products": [
            {'name': 'Neem Oil Spray', 'url': 'https://www.amazon.in/gp/aw/d/B09ZRP5VMM/', 'image_url': '/static/images/neem2.jpeg'},
            {'name': 'Mancozeb 75% WP', 'url': 'https://www.crystalcropprotection.com/cropprotection/detail/blue-copper', 'image_url': '/static/images/blue-copper.jpg'}
        ],
        "Precautions": [
            "Use a hand sprayer to avoid excessive pesticide use",
            "Keep pesticides away from children and animals",
            "Wash hands after spraying"
        ]
    },
    "Powder mildew": {  # Fixed mismatched quotes
        "Symptoms": [
            "White, powdery patches on leaves and stems",
            "Leaves turn yellow and dry out",
            "Slowed growth reduces cotton yield"
        ],
        "reason": [
            "High humidity and warm weather encourage fungal growth",
            "Poor air circulation worsens the problem"
        ],
        "Effects": [
            "Reduces cotton growth and fiber quality",
            "Fewer bolls lead to lower yield"
        ],
        "Treatment": {
            "Organic": [
                "Spray Cow’s Milk Solution (1 part milk, 9 parts water)",
                "Use Sulfur Dust to prevent mildew",
                "Avoid overhead watering"
            ],
            "Chemical": [
                "Wettable Sulfur (2 g per liter of water)",
                "Trifloxystrobin 25% + Tebuconazole 50% WG (1 g per liter of water)"
            ]
        },
        'Products': [
            {'name': 'Sulfur Powder', 'url': 'https://www.amazon.in/SENTALAB-Sulphur-Powder-NO-7704-34-9-Purity/', 'image_url': '/static/images/sulfur.jpg'},
            {'name': 'Propiconazole 25% EC', 'url': 'https://www.bighaat.com/collections/carbendazim-50', 'image_url': '/static/images/sprint.jpg'}
        ],
         "Precautions": [
            "Always use gloves and a mask",
            "Keep children and animals away from the sprayed area",
            "Avoid spraying near water sources"
        ]
    },
    "Healthy leaf": {
        "Symptoms": [
            "Bright green leaves with no discoloration or spots"
        ],
        "reason": [
            "Proper watering, sunlight, and care"
        ],
        "Effects": [
            "Strong plant growth and good fiber yield"
        ],
        "Treatment": "No treatment required. Maintain good care practices."
    },
    "Unknown": {
        "Message": "Unrelated image detected. Please upload an image related to cotton plants."
    }
}




@app.route("/")
def index():
    return jsonify({"status": "success", "message": "Cotton Disease Backend API is running"}), 200


@app.route('/weather', methods=['GET', 'POST'])
def home():
    weather_data = None

    if request.method == 'POST':
        city = request.form.get('city', '').strip()
        if not city:
            return jsonify({"error": "City name is required"}), 400
            
        api_key = "442f888957d18437f16405accc5e5120"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    "city": data["name"],
                    "temperature": data["main"]["temp"],
                    "description": data["weather"][0]["description"].capitalize(),
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "pressure": data["main"]["pressure"],
                    "icon": data["weather"][0]["icon"]
                }
                return jsonify(weather_data), 200
            else:
                return jsonify({"error": "City not found. Please try again."}), 404
        except Exception as e:
            return jsonify({"error": f"Failed to fetch weather: {str(e)}"}), 500
    
    return jsonify({"message": "Send a POST request with city name to get weather data"}), 200


@app.route("/learn")
def learn():
    return jsonify({"message": "Learn endpoint"}), 200

@app.route("/al")
def al():
    return jsonify({"message": "AL endpoint"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        response = {
            "status": "error",
            "message": "No file uploaded. Please upload an image."
        }
        return jsonify(response), 400

    file = request.files['file']
    if not file.filename:
        response = {
            "status": "error",
            "message": "No file selected. Please choose a valid image file."
        }
        return jsonify(response), 400

    try:
        # Ensure directories exist
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(uploads_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        # Save and process the image
        filepath = os.path.join(uploads_dir, file.filename)
        file.save(filepath)

        # Preprocess the image
        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Copy image to static for display
        static_filename = file.filename
        static_path = os.path.join(static_dir, static_filename)
        shutil.copy2(filepath, static_path)

        # Handle unrelated images with a confidence threshold
        if predicted_class == 'Unknown' or confidence < 0.7:
            return jsonify({
                "status": "success",
                "predicted_class": "Unrelated image detected. Please upload an image related to cotton plants.",
                "confidence": confidence,
                "image_url": f"/static/{static_filename}",
                "solutions": {"message": "Unrelated image detected. Please upload an image related to cotton plants."}
            }), 200

        if predicted_class == 'Healthy leaf':
            return jsonify({
                "status": "success",
                "predicted_class": "Healthy Image",
                "confidence": confidence,
                "image_url": f"/static/{static_filename}",
                "solutions": {}
            }), 200

        # Get solutions for the predicted class
        solution = solutions.get(predicted_class, {})
        return jsonify({
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "image_url": f"/static/{static_filename}",
            "solutions": solution
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"An error occurred during prediction: {str(e)}"
        }), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)