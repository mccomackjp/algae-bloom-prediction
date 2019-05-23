#!/bin/bash

# Runs a tensorflow serving docker container.

docker run -p 80:8501 --mount type=bind,source=$(pwd)/../cnn_model,target=/models/cnn_model -e MODEL_NAME=cnn_model -t tensorflow/serving &

