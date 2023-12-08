#!/bin/bash

# Function to handle Ctrl+C
cleanup() {
  echo "Caught Ctrl+C. Stopping the loop."

  # Stop the background process if it's running
  if [ -n "$INFLUXDB_CPU_PID" ]; then
    kill "$INFLUXDB_CPU_PID"
  fi

  # Exit the script
  exit 0
}

# Set the trap to call the cleanup function on Ctrl+C
trap cleanup INT
# Get the time range from the environment variable or use the default value "-1m"
TIME_RANGE=${TIME_RANGE:="-1m"}

n_NEURONS=${n_NEURONS:="100"}

#Train LSTM model
echo "Executing: python3 LSTM.py --n_neurons $n_NEURONS"

python3 LSTM.py --n_neurons="$n_NEURONS"

while true; do


  # Print the command being executed for debugging
  echo "Executing: python3 influxdb_cpu_usage.py &"

  # Run the first script in the background
  python3 influxdb_cpu_usage.py &

  # Capture the PID of influxdb_cpu_usage.py
  INFLUXDB_CPU_PID=$!


  # Print the command being executed for debugging
  echo "Executing: python3 load_model.py --time-range $TIME_RANGE"

  # Run load_model.py with the current time range
  python3 load_model.py --time-range="$TIME_RANGE"

  # Ask the user for input to set a new time range
  read -p "Do you want to set a new time range? Enter the new value (e.g., -5h): " new_time_range

  # Update the TIME_RANGE variable with the new value
  TIME_RANGE="$new_time_range"

  # Optionally, you can ask the user for input or set a specific condition to exit the loop
  # For example, you can use the read command to take user input
  read -p "Do you want to continue? (y/n): " choice
  if [ "$choice" != "y" ]; then
    break
  fi
done



# Stop the background process when the loop exits
cleanup



