from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.predict import predict  # our single source of truth for inference

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# Request schema (same fields you collect in the UI)
class HeartData(BaseModel):

    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def api_predict(data: HeartData):
    try:
        out = predict(data.model_dump())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

# --- Gradio UI wrappers the same predict() ---
def gradio_interface(
    age, 
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
):
    payload = {
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
        "thal": thal,
    }
    out = predict(payload)

    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[

    gr.Number(label="Age"),

    gr.Dropdown([0,1],label="Sex"),

    gr.Dropdown([1  ,2,3,4],label="Chest Pain"),

    gr.Number(label="Resting BP"),

    gr.Number(label="Cholesterol"),

    gr.Dropdown([0,1],label="Fasting Blood Sugar"),

    gr.Dropdown([0,1,2],label="Rest ECG"),

    gr.Number(label="Maximum Heart Rate"),

    gr.Dropdown([0,1],label="Exercise Angina"),

    gr.Number(label="Oldpeak"),

    gr.Dropdown([0,1,2],label="Slope"),

    gr.Dropdown([0,1,2,3],label="CA"),

    gr.Dropdown([3,6,7],label="Thal"),
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Fill in the patient details to get a heart disease prediction.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")
