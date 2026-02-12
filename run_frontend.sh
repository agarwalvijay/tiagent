#!/bin/bash

echo "Starting TI Semiconductor Agent Frontend..."
echo ""

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
    echo ""
fi

# Start frontend
npm start
