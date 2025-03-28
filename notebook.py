import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify, render_template
import xgboost as xgb

# 1. CHARGEMENT DES DONNÉES
df = pd.read_csv("housing_sales_ma_.csv")

df.rename(columns={"price_£": "price_GBP", "proprety type": "property_type"}, inplace=True)
df["price_MAD"] = df["price_GBP"] * 12.5
df.drop(columns=["price_GBP", "address"], inplace=True)
df["principale"] = df.groupby("city")["principale"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))

# 2. EXPLORATION DES DONNÉES
plt.figure(figsize=(10, 5))
sns.histplot(df["price_MAD"], bins=50, kde=True, color="blue")
plt.title("Distribution des prix en MAD")
plt.show()

numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()

#PRÉPARATION DES DONNÉES
X = df.drop(columns=["price_MAD"])
y = np.log(df["price_MAD"])  # Transformation logarithmique pour stabiliser la variance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ["surface", "bedroom", "bathroom"]
cat_features = ["property_type", "city", "principale"]

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

#XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)

#Modèle avec pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])

#Définir les hyperparamètres à tester pour GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [4, 6, 8],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9],
}

#GridSearchCV optimisation des hyperparamètres
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)

#Entraînement avec GridSearch
grid_search.fit(X_train, y_train)

#meilleurs hyperparamètres
print("Meilleurs hyperparamètres :", grid_search.best_params_)

#Prédictions avec le meilleur modèle
y_pred = grid_search.best_estimator_.predict(X_test)
y_test_exp = np.exp(y_test)  # Convertir les valeurs réelles à leur échelle originale
y_pred_exp = np.exp(y_pred)

#Evaluation
print(f"MAE: {mean_absolute_error(y_test_exp, y_pred_exp)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))}")
print(f"R²: {r2_score(y_test, y_pred)}")

#Sauvegarde
joblib.dump(grid_search.best_estimator_, "model/real_estate_model.pkl")


#DÉPLOIEMENT FLASK
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    model = joblib.load("model/real_estate_model.pkl")
    prediction = np.exp(model.predict(input_data)[0])  
    return jsonify({"predicted_price_MAD": prediction})

if __name__ == "__main__":
    app.run(debug=True)
