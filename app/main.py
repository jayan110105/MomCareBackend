import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
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

# Define daily max limits for each nutrient
max_list = [2000, 100, 200, 3500, 325, 40]

# Define Pydantic models for input data
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

class NutritionValues(BaseModel):
    calories: int
    fats: int
    proteins: int
    iron: int
    carbohydrates: int
    fibre: int
    veg_only: bool

# Min-max scaling functions
def min_max_scale(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals), (min_vals, max_vals)

def transform(data, scaler_params):
    min_vals, max_vals = scaler_params
    return [(val - min_val) / (max_val - min_val) for val, min_val, max_val in zip(data, min_vals, max_vals)]

# Nearest neighbors function
def euclidean_distance(vec1, vec2):
    return sum((x - y) ** 2 for x, y in zip(vec1, vec2)) ** 0.5

def find_nearest_neighbors(data, target, n_neighbors=5):
    distances = [(i, euclidean_distance(row, target)) for i, row in enumerate(data)]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    return [index for index, _ in distances[:n_neighbors]]

# Recommendation function
def recommend(dataset, input_data, max_nutritional_values, veg_only=True):
    extracted_data = dataset.copy()
    for column, maximum in zip(extracted_data.columns[1:-1], max_nutritional_values):
        extracted_data = extracted_data[extracted_data[column] <= maximum]

    if veg_only:
        extracted_data = extracted_data[extracted_data['VegNovVeg'] == 0]

    prep_data, scaler_params = min_max_scale(extracted_data.iloc[:, 1:-1].values)
    weighted_input = transform(input_data, scaler_params)

    neighbor_indices = find_nearest_neighbors(prep_data, weighted_input, n_neighbors=5)
    return extracted_data.iloc[neighbor_indices].to_dict(orient="records")

# Define a recommendation endpoint for nutrition
@app.post("/recommendations/")
async def get_recommendations(nutrition_values: NutritionValues):
    input_data = [
        nutrition_values.calories,
        nutrition_values.fats,
        nutrition_values.proteins,
        nutrition_values.iron,
        nutrition_values.carbohydrates,
        nutrition_values.fibre,
    ]
    recommendations = recommend(dataset, input_data, max_list, nutrition_values.veg_only)
    return {"recommendations": recommendations}

# Vercel requires an ASGI app object
app = app
