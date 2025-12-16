import pandas as pd
import time

def read_sensor_data(path="data/sensors_sample.csv", delay_seconds=2):
    """
    Generator that simulates a real-time sensor data stream.
    """
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(delay_seconds)
