from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle sauvegardé
model = joblib.load("model/real_estate_model.pkl")

def predict_price(city, property_type, surface, bedroom, bathroom, principale="Unknown"):
    """
    Prédire le prix immobilier en fonction des variables d'entrée.
    
    Parameters:
        city (str): La ville où se situe la propriété
        property_type (str): Le type de propriété (e.g., 'Apartment', 'Villa', etc.)
        surface (float): La surface de la propriété (en m²)
        bedroom (int): Le nombre de chambres
        bathroom (int): Le nombre de salles de bain
        principale (str): L'indicateur 'principale' (par défaut "Unknown")
    
    Returns:
        float: Le prix estimé en MAD (dirhams marocains)
    """
    input_data = pd.DataFrame([{
        'city': city,
        'property_type': property_type,
        'surface': surface,
        'bedroom': bedroom,
        'bathroom': bathroom,
        'principale': principale,
    }])
    
    prediction = model.predict(input_data)
    predicted_price = np.exp(prediction) / 10  
    return predicted_price[0]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    city = data['city']
    property_type = data['property_type']
    surface = data['surface']
    bedroom = data['bedroom']
    bathroom = data['bathroom']
    
    predicted_price = predict_price(city, property_type, surface, bedroom, bathroom)
    
    # Conversion du résultat en float pour éviter les problèmes de sérialisation
    predicted_price = float(predicted_price)
    
    return jsonify({"predicted_price_MAD": predicted_price})

if __name__ == "__main__":
    app.run(debug=True)
