from influxdb_client import InfluxDBClient, Point, WritePrecision, DeletePredicateRequest
from influxdb_client.client.write_api import SYNCHRONOUS
import psutil
from datetime import datetime
import time, os
import json

# InfluxDB configuration
INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = 'my_dataset'
INFLUXDB_USER = 'jimkap'
INFLUXDB_ORG = 'UP'
INFLUXDB_TOKEN = "BJGdSkXbyCY7h1wjxbPJv2ZzzhdY-AlUyYA1yOlvOrAhyV_ymGq-O3iwX07leV3kClLALhCz0m-p6oWPtKHQmg=="

# Connect to InfluxDB
client = InfluxDBClient(url=f"http://{INFLUXDB_HOST}:{INFLUXDB_PORT}", token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)


# Function to collect and write CPU metrics to InfluxDB
def collect_and_write_cpu_metrics():
    """
        Continuously collects CPU usage metrics using psutil and writes them to InfluxDB.

        Metrics are collected at regular intervals and include the CPU percentage usage.

        """
    try:
        while True:
            # Collect CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.3)

            # Create Point
            point = Point("cpu_usage").tag("host", "localhost").field("cpu_percent", cpu_percent)

            # Write data to InfluxDB
            write_api.write(bucket=INFLUXDB_DATABASE, org=INFLUXDB_ORG, record=point)

    except KeyboardInterrupt:

        print("Script terminated by user.")

    finally:

        client.close()


# Run the script
if __name__ == "__main__":
    collect_and_write_cpu_metrics()
