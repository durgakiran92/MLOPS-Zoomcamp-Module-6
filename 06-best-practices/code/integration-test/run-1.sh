#!/usr/bin/env bash

set -e

# Set defaults if not provided
LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME:-stream-model-duration:latest}
PREDICTIONS_STREAM_NAME=${PREDICTIONS_STREAM_NAME:-ride_predictions}

# Rebuild image if default/placeholder or not found
if [[ "$LOCAL_IMAGE_NAME" == "123" || -z $(docker images -q "$LOCAL_IMAGE_NAME") ]]; then
  echo "🛠️ Building Docker image: $LOCAL_IMAGE_NAME"
  docker build -t "$LOCAL_IMAGE_NAME" ..
else
  echo "✅ Using existing Docker image: $LOCAL_IMAGE_NAME"
fi

# Export vars for docker-compose
export LOCAL_IMAGE_NAME
export PREDICTIONS_STREAM_NAME

# Start services
echo "🚀 Starting containers with docker-compose..."
docker-compose up -d

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready on port 8080..."
for i in {1..20}; do
  if curl -s http://localhost:8080/health | grep -q "OK"; then
    echo "✅ Backend is ready!"
    break
  fi
  sleep 1
done

# Run the integration test
echo "🧪 Running integration test..."
pipenv run python test_docker.py

# Optional: Stop containers after test
# docker-compose down
