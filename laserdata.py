import csv
import sys
from datetime import UTC, datetime, timedelta, timezone

import requests
import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python laserdata.py <password> <session_token>")
        sys.exit(1)
    password = sys.argv[1]
    session_token = sys.argv[2]

    token_response = requests.post(
        "http://iot.multiteknik.dk:8080/api/auth/login",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "username": "chbemo@adm.aau.dk",
            "password": password,
        },
    )
    if token_response.status_code == 401:
        print("401 Error: Invalid password")
        sys.exit(1)
    token_response.raise_for_status()

    date = input("Enter date (DD-MM-YYYY): ")
    try:
        date = datetime.strptime(date, "%d-%m-%Y").date()
    except ValueError:
        print("Invalid date format. Please use DD-MM-YYYY.")
        sys.exit(1)

    token = token_response.json()["token"]

    start = datetime(date.year, date.month, date.day, 0, tzinfo=UTC)
    end = start + timedelta(days=1) - timedelta(hours=1)

    count_map = {}
    for i in range(24):
        count_map[
            datetime(
                date.year, date.month, date.day, 1, tzinfo=timezone(timedelta(hours=2))
            )
            + timedelta(hours=i)
        ] = {"innhabit": 0, "laser": 0}

    response = requests.get(
        "http://iot.multiteknik.dk:8080/api/plugins/telemetry/DEVICE/47afeb80-276e-11ec-92de-537d4a380471/values/timeseries",
        params={
            "keys": "c1,c2,c3",
            "startTs": str(int(start.timestamp() * 1000)),
            "endTs": str(int(end.timestamp() * 1000)),
            "limit": 1000,
        },
        headers={"X-Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()

    json = response.json()
    c1, c2, c3 = np.array(json["c1"]), np.array(json["c2"]), np.array(json["c3"])
    laser_counts = []
    for x, y, z in zip(c1, c2, c3):
        assert x["ts"] == y["ts"]
        assert y["ts"] == z["ts"]
        value = int(x["value"])  # + int(y["value"]) + int(z["value"])
        laser_counts.append({"ts": x["ts"], "value": value})
    laser_counts.sort(key=lambda x: x["ts"])
    for count in laser_counts:
        timestamp = datetime.fromtimestamp(
            count["ts"] // 1000, tz=timezone(timedelta(hours=2))
        )
        if timestamp in count_map:
            count_map[timestamp]["laser"] = int(count["value"]) * 2

    response = requests.get(
        "https://innhabit.dk/export/csv/",
        cookies={
            "sessionid": session_token,
            "tzinfo": "Europe/Copenhagen",
        },
        params={
            "from_date": date,
            "to_date": (
                date + timedelta(days=1) if date != datetime.now().date() else date
            ),
            "entrances": [1, 2, 3],
        },
    )
    if response.status_code == 403:
        print("403 Error: Invalid session token")
        sys.exit(1)
    response.raise_for_status()
    reader = csv.reader(response.text.splitlines())
    next(reader)
    window_end = start + timedelta(hours=1)
    event_count = 0
    for row in reader:
        timestamp = datetime.fromisoformat(row[1])
        while timestamp > window_end:
            if window_end in count_map:
                count_map[window_end]["innhabit"] = event_count
            window_end = window_end + timedelta(hours=1)
            event_count = 0
        event_count += 1

    laser = []
    innhabit = []
    labels = []
    while count_map:
        key = min(count_map.keys())
        items = count_map.pop(key)
        laser.append(items["laser"])
        innhabit.append(items["innhabit"])
        labels.append(key.strftime("%H"))
    plt.plot(labels, innhabit, label="INNHABIT")
    plt.plot(labels, laser, label="Laser")
    plt.xlabel("Time of day")
    plt.ylabel("Number of events measured")
    plt.legend()
    plt.show()
    # plt.savefig("INNHABIT_VS_LASER.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
