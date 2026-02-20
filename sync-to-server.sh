#!/bin/bash

# Sync TI Agent to production server
SERVER="vagarwal@tiagent.theagarwals.com"
REMOTE_DIR="/home/vagarwal/ti-agent"

echo "🔄 Syncing TI Agent to $SERVER:$REMOTE_DIR"
echo ""

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

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps:"
echo "1. SSH into server: ssh $SERVER"
echo "2. Navigate to app: cd $REMOTE_DIR"
echo "3. Run deployment (creates venv & installs deps): ./deploy.sh"
echo ""
echo "That's it! The deploy script will:"
echo "  ✓ Create Python venv"
echo "  ✓ Install dependencies"
echo "  ✓ Set up systemd service"
echo "  ✓ Configure nginx"
