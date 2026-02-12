#!/bin/bash

echo "Starting TI Semiconductor Agent Backend..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Check if ChromaDB is populated
if [ ! -d "chroma_db" ]; then
    echo "⚠️  Warning: chroma_db directory not found"
    echo "Run ./setup.sh to ingest datasheets first"
    exit 1
fi

# Start backend from project root (so .env is found)
python3 -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
