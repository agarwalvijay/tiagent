#!/bin/bash

echo "🚀 Deploying TI Agent to tiagent.theagarwals.com"
echo ""

SERVER="vagarwal@tiagent.theagarwals.com"
REMOTE_DIR="/home/vagarwal/tiagent/tiagent"

# 1. Build frontend locally
echo "📦 Building frontend locally..."
cd frontend
npm install
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

cd ..

# 2. Sync to server (excluding source files, including build)
echo ""
echo "📤 Syncing to server..."
rsync -avz --progress \
  --exclude 'node_modules/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'frontend/src/' \
  --exclude 'frontend/public/' \
  --exclude 'frontend/node_modules/' \
  --exclude 'venv/' \
  --exclude 'env/' \
  --exclude '.venv/' \
  --exclude '*.log' \
  --exclude '.npm/' \
  --exclude 'Datasheets/' \
  --exclude 'backend/vectorstore/' \
  --exclude '.env' \
  --exclude '.claude/' \
  --delete \
  . "$SERVER:$REMOTE_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Sync failed!"
    exit 1
fi

# 3. Deploy on server
echo ""
echo "⚙️  Running deployment on server..."
ssh "$SERVER" "cd $REMOTE_DIR && ./deploy-pm2.sh"

if [ $? -ne 0 ]; then
    echo "❌ Deployment failed!"
    exit 1
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌍 Your app: http://tiagent.theagarwals.com"
echo ""
echo "Check status:"
echo "  ssh $SERVER 'pm2 status'"
echo "  ssh $SERVER 'pm2 logs'"
