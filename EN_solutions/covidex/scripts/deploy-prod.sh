#!/bin/bash

sh ./scripts/update-client.sh

# run server without the development flag
cd api
pip install -r requirements.txt
export DEVELOPMENT=False

# kill running servers
echo "Stopping existing servers..."
pkill -9 uvicorn
sleep 10 # Buffer to make sure servers stop

echo "Starting server..."
PORT=${PORT:-8000}
nohup uvicorn app.main:app --port=$PORT --host 0.0.0.0 &

echo "Waiting for server availability..."
status_code=$(curl --write-out %{http_code} --silent --output /dev/null http://localhost:$PORT/api/status)
while [ "$status_code" -ne 200 ]; do
  echo "Server not available, trying again in 10 seconds..."
  sleep 10
  status_code=$(curl --write-out %{http_code} --silent --output /dev/null http://localhost:$PORT/api/status)
done

echo "Server started successfully! Logs are available at api/logs/"
