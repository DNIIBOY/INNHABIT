import csv
import sys
from datetime import datetime, timedelta, timezone

import requests
from matplotlib import pyplot as plt


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python occupancydata.py <session_token>")
        sys.exit(1)
    session_token = sys.argv[1]

    date = input("Enter date (DD-MM-YYYY): ")
    try:
        date = datetime.strptime(date, "%d-%m-%Y").date()
    except ValueError:
        print("Invalid date format. Please use DD-MM-YYYY.")
        sys.exit(1)

    system_response = requests.get(
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
        },
    )
    system_response.raise_for_status()

    manual_response = requests.get(
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
            "test_events": True,
        },
    )
    manual_response.raise_for_status()

    tz = timezone(timedelta(hours=2))

    count_map = {}
    interval_size = 5
    intervals = 7 * (60 // interval_size)
    for i in range(0, intervals * interval_size, interval_size):
        count_map[
            datetime(date.year, date.month, date.day, 10, 0, 0, tzinfo=tz)
            + timedelta(minutes=i)
    ] = {"system": None, "manual": None}

    reader = csv.reader(system_response.text.splitlines())
    next(reader)
    start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    window_end = start + timedelta(minutes=interval_size)
    current_count = 0
    labels = [window_end.strftime("%H:%M")]
    for row in reader:
        timestamp = datetime.fromisoformat(row[1])
        if timestamp < min(count_map.keys()):
            continue
        if timestamp > end:
            break
        while timestamp > window_end:
            if window_end in count_map:
                count_map[window_end]["system"] = current_count
            window_end = window_end + timedelta(minutes=interval_size)
        if row[2] == "Ind":
            current_count += 1
        elif row[2] == "Ud":
            current_count -= 1

    reader = csv.reader(manual_response.text.splitlines())
    next(reader)
    start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    window_end = start + timedelta(minutes=interval_size)
    current_count = 0
    labels = [window_end.strftime("%H:%M")]
    for row in reader:
        timestamp = datetime.fromisoformat(row[1])
        if timestamp > end:
            break
        while timestamp > window_end:
            if window_end in count_map:
                count_map[window_end]["manual"] = current_count
            window_end = window_end + timedelta(minutes=interval_size)
        if row[2] == "Ind":
            current_count += 1
        elif row[2] == "Ud":
            current_count -= 1

    innhabit = []
    manual = []
    labels = []
    prev_inn = None
    prev_manual = None
    while count_map:
        key = min(count_map.keys())
        items = count_map.pop(key)
        inn, man = items["system"], items["manual"]
        if inn is None:
            if prev_inn is None:
                prev_inn = 0
            inn = prev_inn
        prev_inn = inn
        if man is None:
            if prev_manual is None:
                prev_manual = 0
            man = prev_manual
        prev_manual = man
        innhabit.append(inn)
        manual.append(man)
        labels.append(key.strftime("%H:%M"))

    plt.plot(labels, innhabit, label="INNHABIT")
    plt.plot(labels, manual, label="Manual", linestyle="--", color="#2CA02C")
    ax = plt.gca()

    tick_positions = []
    tick_labels = []
    # Add positions and labels only for times ending with :00
    for i, label in enumerate(labels):
        if label.endswith(":00"):
            tick_positions.append(i)
            tick_labels.append(label)

    # Set the x-ticks and x-labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Rotate labels for better readability
    plt.xticks(rotation=45)

    plt.legend()
    # plt.title(date.strftime("%d/%m %Y"))
    plt.xlabel("Time of day")
    plt.ylabel("Current occupants in building")
    plt.tight_layout()
    # plt.show()
    plt.savefig("INNHABIT_manual_occupancy.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
