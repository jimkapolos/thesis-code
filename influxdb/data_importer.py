from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd

# InfluxDB configuration
INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = 'my_dataset'
INFLUXDB_USER = 'jimkap'
INFLUXDB_ORG = 'UP'
INFLUXDB_TOKEN = "BJGdSkXbyCY7h1wjxbPJv2ZzzhdY-AlUyYA1yOlvOrAhyV_ymGq-O3iwX07leV3kClLALhCz0m-p6oWPtKHQmg=="

# Connect to InfluxDB
client = InfluxDBClient(url=f"http://{INFLUXDB_HOST}:{INFLUXDB_PORT}", token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# Function to query and return CPU usage values as a Pandas Series
def query_cpu_metrics():
    """
    Queries CPU usage metrics from InfluxDB and returns a Pandas Series.
    """
    try:
        # InfluxDB Flux query to select CPU metrics
        flux_query = '''
        from(bucket: "my_dataset")
          |> range(start: -10m)
          |> filter(fn: (r) => r["_measurement"] == "cpu_usage")
          |> filter(fn: (r) => r["_field"] == "cpu_percent")
          |> filter(fn: (r) => r["host"] == "localhost")
          |> yield(name: "mean")
        '''
        # Execute the query
        result = query_api.query(org=INFLUXDB_ORG, query=flux_query)

        # Extract CPU usage values from records
        cpu_usages = [record.values['_value'] for table in result for record in table.records]

        # Create Pandas Series
        series = pd.Series(cpu_usages, name='cpu_percent')

        return series

    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.Series()

    finally:
        client.close()

# Run the script
if __name__ == "__main__":
    cpu_metrics_series = query_cpu_metrics()

    # Display the Series
    print(cpu_metrics_series)
