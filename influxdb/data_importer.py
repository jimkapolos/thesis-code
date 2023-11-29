from influxdb_client import InfluxDBClient
import pandas as pd


# Function to connect to InfluxDB and create query API
def connect_to_influxdb(INFLUXDB_HOST, INFLUXDB_PORT, INFLUXDB_ORG, INFLUXDB_TOKEN):
    # Connect to InfluxDB
    client = InfluxDBClient(url=f"http://{INFLUXDB_HOST}:{INFLUXDB_PORT}", token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    return query_api


# Function to query and return CPU usage values as a Pandas Series
def query_cpu_metrics(query_api, INFLUXDB_ORG):
    """
    Queries CPU usage metrics from InfluxDB and returns a Pandas Series.
    """
    # InfluxDB Flux query to select CPU metrics
    flux_query = '''
        from(bucket: "my_dataset")
          |> range(start: -1m)
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
    series = pd.Series(cpu_usages, name='cpu_usage')
    return series



