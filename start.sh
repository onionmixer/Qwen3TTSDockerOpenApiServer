#!/bin/bash
cd "$(dirname "$0")"
docker compose up --build -d
echo ""
echo "Waiting for server to start..."
echo "Use 'docker compose logs -f' to watch logs"
