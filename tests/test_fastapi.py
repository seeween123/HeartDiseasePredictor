import requests

url = "http://127.0.0.1:8000/predict"

sample_data = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}

response = requests.post(url, json=sample_data)
print("Status Code:", response.status_code)
print("Response:", response.json())
