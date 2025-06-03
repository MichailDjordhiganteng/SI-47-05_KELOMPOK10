from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model dan scaler
model = joblib.load("model_regresi.pkl")
scaler = joblib.load("scaler.pkl")  # Pastikan file scaler.pkl juga ada

# Schema untuk input
class SalesInput(BaseModel):
    quantity: float
    unit_price: float

@app.post("/predict")
def predict_sales(data: SalesInput):
    input_array = np.array([[data.quantity, data.unit_price]])
    scaled_input = scaler.transform(input_array)  # penting!
    prediction = model.predict(scaled_input)[0]
    return {"predicted_sales": round(prediction, 2)}
