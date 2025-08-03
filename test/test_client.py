import requests

test_data = {
    "landmarks": [[[0.5, 0.5, 0.5]]*33]*5,
    "muscle_group": "BACK",
    "exercise": "deadlifts",
    "timestamp": 0.0
}

response = requests.post("http://localhost:8000/process-frame", json=[test_data])
print("Response:", response.json())