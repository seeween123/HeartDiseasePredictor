"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Heart Disease Prediction model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.predict import predict  # Core ML inference logic
import uvicorn

# Initialize FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML API for predicting heart disease in patients based on clinical features. Provides both REST API and Gradio UI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/health")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class PatientData(BaseModel):
    """
    Patient data schema for heart disease prediction.
    
    This schema defines the exact features required for heart disease prediction.
    All features match the original dataset structure for consistency.
    """

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: PatientData):
    """
    Main prediction endpoint for heart disease prediction.

    This endpoint:
    1. Receives validated patient data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns heart disease prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to have heart disease"} or {"prediction": "Not likely to have heart disease"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching PatientData schema
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all patient features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label="age", value=65, minimum=0, maximum=100),
        gr.Dropdown(["0", "1"], label="sex", value="1"),
        gr.Dropdown(["1", "2", "3", "4"], label="cp", value="1"),
        gr.Number(label="trestbps", value=145, minimum=50, maximum=300),
        gr.Number(label="chol", value=233, minimum=50, maximum=700),
        gr.Dropdown(["0", "1"], label="fbs", value="1"),
        gr.Dropdown(["0", "1", "2"], label="restecg", value="2"),
        gr.Number(label="thalach", value=150, minimum=10, maximum=250),
        gr.Dropdown(["0", "1"], label="exang", value="0"),
        gr.Number(label="oldpeak", value=2.3, minimum=0, maximum=10),
        gr.Dropdown(["1", "2", "3"], label="slope", value="3"),
        gr.Dropdown(["0", "1", "2", "3"], label="ca", value="0"),
        gr.Dropdown(["3", "6", "7"], label="thal", value="6"),
    ],
    outputs=gr.Textbox(label="Heart Disease Prediction", lines=2),
    title="🔮 Heart Disease Predictor",
    description="""
    **Predict heart disease probability using machine learning**
    
    Fill in the patient details below to get a heart disease prediction. The model uses XGBoost trained on 

    historical patient data to identify individuals at risk of developing heart disease.
    
    💡 **Tip**: Older patients with higher cholesterol levels and certain chest pain types 
    tend to have a higher risk of heart disease.
    """,
       #theme=gr.themes.Soft()  # Professional appearance
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )