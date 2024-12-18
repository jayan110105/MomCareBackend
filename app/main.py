import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import joblib

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Replace with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the dataset for dietary recommendations
data_path = os.path.join(os.path.dirname(__file__), "unique_indian_foods_dataset.csv")
data = pd.read_csv(data_path, encoding="ISO-8859-1")
columns = ['Food_items', 'Calories', 'Fats', 'Proteins', 'Iron', 'Carbohydrates', 'Fibre', 'VegNovVeg']
dataset = data[columns].copy()

# Load the trained models
rf_model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl")
xgb_model_path = os.path.join(os.path.dirname(__file__), "xgb_model.pkl")
random_forest_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)

# Define daily max limits for each nutrient
max_list = [2000, 100, 200, 3500, 325, 40]

# Define Pydantic models for input data
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

class PredictionInput(BaseModel):
    Age: str
    FeelingSadOrTearful: str
    IrritableTowardsBabyAndPartner: str
    TroubleSleepingAtNight: str
    ProblemsConcentratingOrMakingDecision: str
    OvereatingOrLossOfAppetite: str
    FeelingOfGuilt: str
    ProblemsOfBondingWithBaby: str
    SuicideAttempt: str

class NutritionValues(BaseModel):
    calories: int
    fats: int
    proteins: int
    iron: int
    carbohydrates: int
    fibre: int
    veg_only: bool  # User input for vegetarian preference

# Function to extract data based on user-defined limits
def extract_data(dataset, max_nutritional_values, veg_only=True):
    extracted_data = dataset.copy()
    for column, maximum in zip(extracted_data.columns[1:-1], max_nutritional_values):
        extracted_data = extracted_data[extracted_data[column] <= maximum]

    if veg_only:
        extracted_data = extracted_data[extracted_data['VegNovVeg'] == 0]  # Assuming 0 is for vegetarian

    return extracted_data

# Function to scale the data
def scaling(extracted_data):
    scaler = MinMaxScaler()
    prep_data = scaler.fit_transform(extracted_data.iloc[:, 1:-1].to_numpy())
    return prep_data, scaler

# Recommendation function
def recommend(dataset, input_data, max_nutritional_values, veg_only=True):
    extracted_data = extract_data(dataset, max_nutritional_values, veg_only)
    prep_data, scaler = scaling(extracted_data)

    # Weights can be adjusted based on your requirements
    weights = np.array([1.2, 1.0, 2.0, 1.5, 1.0, 1.2])
    weighted_input = scaler.transform(input_data.reshape(1, -1)) * weights

    neigh = NearestNeighbors(metric='euclidean', algorithm='brute')
    neigh.fit(prep_data * weights)  # Weighted data
    distances, neighbor_indices = neigh.kneighbors(weighted_input, n_neighbors=5)

    return extracted_data.iloc[neighbor_indices[0]].to_dict(orient='records')

# Define a prediction endpoint for the random forest model
@app.post("/predict_risk")
async def predict_random_forest(input_data: ModelInput):
    try:
        # Extract features from the input data
        input_features = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4/18, input_data.feature5]])

        # Make a prediction
        prediction = random_forest_model.predict(input_features)

        # Return the prediction as a string
        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

# Define a prediction endpoint for the XGBoost model
@app.post("/predict_anxiety")
async def predict_xgb(input: PredictionInput):
    # Convert input to numerical values
    age_mapping = {'25-30': 1, '30-35': 2, '35-40': 3, '40-45': 4, '45-50': 5}
    feeling_mapping = {'Yes': 2, 'Sometimes': 1, 'No': 0}
    irritable_mapping = {'Yes': 2, 'Sometimes': 1, 'No': 0}
    sleeping_mapping = {'Yes': 2, 'Two or more days a week': 1, 'No': 0}
    conc_mapping = {'Yes': 2, 'Often': 1, 'No': 0}
    overeating_mapping = {'Yes': 2, 'No': 1, 'Not at all': 0}
    guilt_mapping = {'Yes': 2, 'Maybe': 1, 'No': 0}
    bonding_mapping = {'Yes': 2, 'Sometimes': 1, 'No': 0}
    suicide_mapping = {'Yes': 1, 'No': 0}

    input_array = np.array([
        age_mapping[input.Age],
        feeling_mapping[input.FeelingSadOrTearful],
        irritable_mapping[input.IrritableTowardsBabyAndPartner],
        sleeping_mapping[input.TroubleSleepingAtNight],
        conc_mapping[input.ProblemsConcentratingOrMakingDecision],
        overeating_mapping[input.OvereatingOrLossOfAppetite],
        guilt_mapping[input.FeelingOfGuilt],
        bonding_mapping[input.ProblemsOfBondingWithBaby],
        suicide_mapping[input.SuicideAttempt]
    ]).reshape(1, -1)

    # Make prediction
    prediction = xgb_model.predict(input_array)[0]

    return {"prediction": "Feeling anxious" if prediction == 1 else "Not feeling anxious"}

# Define a recommendation endpoint for nutrition
@app.post("/recommendations/")
async def get_recommendations(nutrition_values: NutritionValues):
    input_data = np.array([
        nutrition_values.calories,
        nutrition_values.fats,
        nutrition_values.proteins,
        nutrition_values.iron,
        nutrition_values.carbohydrates,
        nutrition_values.fibre,
    ])
    recommendations = recommend(dataset, input_data, max_list, nutrition_values.veg_only)  # Pass veg_only to the recommend function
    return {"recommendations": recommendations}

# Vercel requires an ASGI app object
app = app
