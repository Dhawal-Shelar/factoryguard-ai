import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

NUM_MACHINES = 500
DAYS = 30
HOURS = DAYS * 24

data = []

start_time = datetime(2024, 1, 1)

for machine_id in range(1, NUM_MACHINES + 1):

    base_temp = np.random.normal(60, 5)
    base_vibration = np.random.normal(0.5, 0.1)
    base_pressure = np.random.normal(30, 3)


    machine_fails = np.random.rand() < 0.01

    if machine_fails:
        failure_hour = np.random.randint(24, HOURS)
    else:
        failure_hour = -1

    for hour in range(HOURS):

        timestamp = start_time + timedelta(hours=hour)

        temp = base_temp + np.random.normal(0, 0.5)
        vibration = base_vibration + np.random.normal(0, 0.02)
        pressure = base_pressure + np.random.normal(0, 1)

        failure = 0


        if machine_fails and hour >= failure_hour - 24 and hour < failure_hour: # error line
            temp += np.random.normal(5, 1)
            vibration += np.random.normal(0.3, 0.05)
            pressure += np.random.normal(5, 1)

        # failure point
        if machine_fails and hour == failure_hour:
            failure = 1

        data.append([
            machine_id,
            timestamp,
            temp,
            vibration,
            pressure,
            failure
        ])

df = pd.DataFrame(data, columns=[
    "machine_id",
    "timestamp",
    "temperature",
    "vibration",
    "pressure",
    "failure"
])

df.to_csv("data/iot_sensor_data.csv", index=False)

print("Dataset successfully generated!")
print("Total rows:", len(df))
print("Failure counts:\n", df["failure"].value_counts())
