#!/bin/bash

BASE_DIR=`pwd`
echo $BASE_DIR
IMAGE_NAME="frontend:latest"
CONTAINER_NAME="frontend_service"
DOCKERFILE_PATH="$BASE_DIR/src/frontend/Dockerfile"
ENV_FILE="$BASE_DIR/.env"
APP_PORT=8502

echo "Building Docker image..."
docker build -t $IMAGE_NAME -f "$DOCKERFILE_PATH" "$BASE_DIR"

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing container $CONTAINER_NAME..."
    docker rm -f $CONTAINER_NAME
fi

echo "Running container $CONTAINER_NAME..."
docker run -d \
    --name $CONTAINER_NAME \
    --env-file "$ENV_FILE" \
    -p $APP_PORT:8502 \
    $IMAGE_NAME

echo "Container $CONTAINER_NAME is running on port $APP_PORT"
