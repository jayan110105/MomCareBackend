# MomCare Backend

This project serves as the backend for the MomCare application, providing machine learning-powered predictions and recommendations related to maternal health. It is built with FastAPI.

## Project Structure

```
ML/
|-- app/
|   `-- main.py             # FastAPI application logic and API endpoints
|-- random_forest_model.pkl # Trained model for risk prediction
|-- xgb_model.pkl           # Trained model for anxiety prediction
|-- unique_indian_foods_dataset.csv # Dataset for dietary recommendations
|-- requirements.txt        # Python dependencies
`-- README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jayan110105/MomCareBackend
    cd ML
    ```

2.  **Create a virtual environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install all the required Python packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the FastAPI application locally, use `uvicorn`. It will start a local server, and the API will be accessible.

```bash
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## API Endpoints

The following endpoints are available:

### 1. Risk Prediction

-   **Endpoint:** `/predict_risk`
-   **Method:** `POST`
-   **Description:** Predicts a general risk level based on input features.
-   **Request Body:**
    ```json
    {
      "feature1": 0.0,
      "feature2": 0.0,
      "feature3": 0.0,
      "feature4": 0.0,
      "feature5": 0.0
    }
    ```
-   **Response:**
    ```json
    {
      "prediction": "some_risk_level"
    }
    ```

### 2. Anxiety Prediction

-   **Endpoint:** `/predict_anxiety`
-   **Method:** `POST`
-   **Description:** Predicts the level of anxiety based on a series of questions.
-   **Request Body:**
    ```json
    {
      "Age": "25-30",
      "FeelingSadOrTearful": "Yes",
      "IrritableTowardsBabyAndPartner": "Yes",
      "TroubleSleepingAtNight": "Yes",
      "ProblemsConcentratingOrMakingDecision": "Yes",
      "OvereatingOrLossOfAppetite": "Yes",
      "FeelingOfGuilt": "Yes",
      "ProblemsOfBondingWithBaby": "Yes",
      "SuicideAttempt": "No"
    }
    ```
-   **Response:**
    ```json
    {
      "prediction": "Feeling anxious"
    }
    ```

### 3. Dietary Recommendations

-   **Endpoint:** `/recommendations/`
-   **Method:** `POST`
-   **Description:** Provides food recommendations based on nutritional requirements.
-   **Request Body:**
    ```json
    {
      "calories": 1800,
      "fats": 80,
      "proteins": 150,
      "iron": 20,
      "carbohydrates": 300,
      "fibre": 30,
      "veg_only": true
    }
    ```
-   **Response:** A JSON object containing a list of recommended food items.
    ```json
    {
      "recommendations": [
        {
          "Food_items": "...",
          "Calories": "...",
          "Fats": "...",
          "Proteins": "...",
          "Iron": "...",
          "Carbohydrates": "...",
          "Fibre": "...",
          "VegNovVeg": "..."
        }
      ]
    }
    ``` 