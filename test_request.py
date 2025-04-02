import requests

url = "http://127.0.0.1:8000/predict"
data = {"data": [9.3252, 41.0, 6.9841, 1187.0, 293.0, 240.0, 113.0, 4.526]}

response = requests.post(url, json=data)
print(response.json())