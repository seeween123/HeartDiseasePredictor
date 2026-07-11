## Heart Disease Prediction – End-to-End Machine Learning Project
### Purpose

Build and deploy an end-to-end machine learning system that predicts
the presence of heart disease using the UCI Heart Disease dataset.

### Problem solved & benefits

Early identification of patients at risk of cardiovascular disease.

Clinical decision support.

REST API for inference.

MLflow experiment tracking.

Docker deployment.

FastAPI service.

Interactive Gradio dashboard.

CI/CD using GitHub Actions.


### Details

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST)
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.
- CI/CD: GitHub Actions builds the image and pushes to Docker Hub. Local or AWS ECS 
- Orchestration: Local / AWS
- Networking: Local / AWS
- Security: Local / AWS
- Observability: Local 

### Deployment flow (high-level)

- Push to main → GitHub Actions builds the Docker image and pushes it to Docker Hub.
- Integrate AWS ECS service is updated (manually or via the workflow) to force a new deployment.
- Users call POST /predict or open the Gradio UI at /ui.
