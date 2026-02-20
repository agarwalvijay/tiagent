#!/bin/bash

echo "🚀 Deploying TI Agent with PM2"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "Creating template .env file..."
    cat > .env << 'EOF'
# LLM Provider (openai, groq, deepseek, google)
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_key_here

# Embeddings
EMBEDDING_PROVIDER=openai

# Backend
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Frontend
FRONTEND_URL=http://tiagent.theagarwals.com
EOF
    echo ""
    echo "❌ Please edit .env file with your actual API keys before continuing!"
    echo "   nano .env"
    exit 1
fi

# 1. Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create venv. Trying with --system-site-packages..."
        python3 -m venv --system-site-packages venv
    fi
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies (CPU-only, no CUDA/nvidia)..."
pip install -r requirements-cpu.txt

# 2. Build frontend
echo ""
echo "📦 Building frontend for production..."
cd frontend
npm install
npm run build
cd ..

# 3. Create logs directory
echo ""
echo "📁 Creating logs directory..."
mkdir -p /home/vagarwal/tiagent/logs

# 4. Stop existing pm2 processes
echo ""
echo "🛑 Stopping existing PM2 processes..."
pm2 stop ecosystem.config.js 2>/dev/null || true
pm2 delete ecosystem.config.js 2>/dev/null || true

# 5. Start with PM2
echo ""
echo "▶️  Starting applications with PM2..."
pm2 start ecosystem.config.js

# 6. Save PM2 configuration
echo ""
echo "💾 Saving PM2 configuration..."
pm2 save

# 7. Set up PM2 to start on boot
echo ""
echo "🔄 Setting up PM2 startup script..."
pm2 startup | grep "sudo" | bash

# 8. Configure nginx
echo ""
echo "🌐 Configuring nginx..."
sudo cp nginx-tichat.conf /etc/nginx/sites-available/tiagent.theagarwals.com
sudo ln -sf /etc/nginx/sites-available/tiagent.theagarwals.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 9. Show status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "PM2 Status:"
pm2 status
echo ""
echo "Nginx status:"
sudo nginx -t
echo ""
echo "🌍 Your app should be available at: http://tiagent.theagarwals.com"
echo ""
echo "Useful commands:"
echo "  PM2 logs:    pm2 logs"
echo "  PM2 status:  pm2 status"
echo "  PM2 restart: pm2 restart ecosystem.config.js"
echo "  PM2 stop:    pm2 stop ecosystem.config.js"
echo "  Nginx logs:  sudo tail -f /var/log/nginx/access.log"
