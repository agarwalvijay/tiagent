#!/bin/bash

echo "======================================"
echo "TI Semiconductor Agent Setup"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Press Enter after you've added your API key..."
fi

# Verify API key is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ Error: OPENAI_API_KEY not set in .env"
    echo "Please add your OpenAI API key to the .env file and run this script again"
    exit 1
fi

echo "✓ .env file configured"
echo ""

# Install backend dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✓ Python dependencies installed"
echo ""

# Ingest datasheets
echo "📄 Ingesting datasheets into ChromaDB..."
echo "This will process 28 TI datasheets (takes 10-15 minutes)..."
echo ""

python3 -m backend.ingestion.ingest_datasheets --datasheet-dir Datasheets

if [ $? -ne 0 ]; then
    echo "❌ Failed to ingest datasheets"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend (in terminal 1):"
echo "   cd backend"
echo "   uvicorn app.main:app --reload"
echo ""
echo "2. Start the frontend (in terminal 2):"
echo "   cd frontend"
echo "   npm install"
echo "   npm start"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
