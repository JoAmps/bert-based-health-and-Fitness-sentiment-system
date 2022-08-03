from requests import request
import requests


r = requests.get("http://127.0.0.1:5000/predict?text=I highly recommend this app to everyone trying to lose weight, excellent app").content
print(r)