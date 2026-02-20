#!/bin/bash

echo "🚀 Deploying TI Agent to tiagent.theagarwals.com"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 0. Set up Python virtual environment
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

echo "Installing Python dependencies..."
pip install -r requirements.txt

# 1. Build frontend (force fresh build for production)
echo "📦 Building frontend for production..."
cd frontend
npm install
npm run build
cd ..

# 2. Set up systemd service
echo "⚙️  Setting up backend service..."
sudo cp ti-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ti-agent
sudo systemctl restart ti-agent

# 3. Set up nginx
echo "🌐 Configuring nginx..."
sudo cp nginx-tichat.conf /etc/nginx/sites-available/tiagent.theagarwals.com
sudo ln -sf /etc/nginx/sites-available/tiagent.theagarwals.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 4. Check status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "Backend status:"
sudo systemctl status ti-agent --no-pager | head -10
echo ""
echo "Nginx status:"
sudo nginx -t
echo ""
echo "🌍 Your app should be available at: http://tiagent.theagarwals.com"
echo ""
echo "Logs:"
echo "  Backend: sudo journalctl -u ti-agent -f"
echo "  Nginx: sudo tail -f /var/log/nginx/access.log"
