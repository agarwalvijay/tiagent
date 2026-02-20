#!/bin/bash

echo "🔧 Preparing TI Agent for deployment..."
echo ""

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $BRANCH"
echo ""

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status -s
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Build frontend
echo "📦 Building frontend for production..."
cd frontend
npm install
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

cd ..

echo ""
echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "1. Review DEPLOYMENT.md for full instructions"
echo "2. Run: ./sync-to-server.sh"
echo "3. SSH to server and run: ./deploy.sh"
echo ""
echo "Quick deploy:"
echo "  ./sync-to-server.sh && ssh vagarwal@tiagent.theagarwals.com 'cd ti-agent && ./deploy.sh'"
