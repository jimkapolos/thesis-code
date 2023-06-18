#!/bin/bash

#./certificates/generate.sh

echo "Starting server"

LOCAL_EPOCHS=3
NEURONS=5
FL_Rounds=2
python server.py --rounds=$FL_Rounds --n_neurons=$NEURONS &
sleep 3  # Sleep for 3s to give the server enough time to start

# Ensure that the Keras dataset used in client.py is already cached.
#python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

for i in `seq 0 21`; do # CHANGED from 0 9 to 0 1
    echo "Starting client $i"
    python client.py --partition=${i} --n_neurons=$NEURONS --n_epochs=$LOCAL_EPOCHS &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
