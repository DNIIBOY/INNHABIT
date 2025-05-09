from datetime import datetime

import requests

token_response = requests.post(
    "http://iot.multiteknik.dk:8080/api/auth/login",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    json={
        "username": "chbemo@adm.aau.dk",
        "password": "BIGSECRET",
    },
)
token_response.raise_for_status()

token = token_response.json()["token"]
print(token)

start = int(datetime(2025, 5, 9, 9).timestamp() * 1000)
end = int(datetime(2025, 5, 9, 10).timestamp() * 1000)

response = requests.get(
    "http://iot.multiteknik.dk:8080/api/plugins/telemetry/DEVICE/47afeb80-276e-11ec-92de-537d4a380471/values/timeseries",
    params={
        "keys": "c1,c2,c3",
        "startTs": str(start),
        "endTs": str(end),
        "limit": 1000,
    },
    headers={"X-Authorization": f"Bearer {token}"},
)

print(response.json())
