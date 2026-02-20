# TI Agent Deployment with PM2

Quick guide for deploying to **tiagent.theagarwals.com** using PM2

## Prerequisites

- Code already pulled to `/home/vagarwal/tiagent/tiagent`
- PM2 installed globally: `npm install -g pm2`
- Python 3.9+, Node.js 16+, nginx installed

## Deployment Steps

### 1. SSH to Server
```bash
ssh vagarwal@tiagent.theagarwals.com
cd /home/vagarwal/tiagent/tiagent
```

### 2. Create .env File
```bash
nano .env
```

Add your configuration:
```env
# LLM Provider
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI (for embeddings)
OPENAI_API_KEY=your_actual_openai_key

# Embeddings
EMBEDDING_PROVIDER=openai

# Backend
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Frontend
FRONTEND_URL=http://tiagent.theagarwals.com
```

### 3. Run Deployment Script
```bash
./deploy-pm2.sh
```

This will:
- ✅ Create Python venv and install dependencies
- ✅ Build frontend for production
- ✅ Start backend and frontend with PM2
- ✅ Configure nginx
- ✅ Set up PM2 auto-restart on boot

### 4. Verify Deployment
```bash
# Check PM2 status
pm2 status

# View logs
pm2 logs

# Test backend
curl http://localhost:8000/api/health

# Test nginx
sudo nginx -t
```

### 5. Set Up SSL (One-time)
```bash
sudo certbot --nginx -d tiagent.theagarwals.com
```

## PM2 Management

### View Status
```bash
pm2 status
```

### View Logs
```bash
# All apps
pm2 logs

# Specific app
pm2 logs ti-agent-backend
pm2 logs ti-agent-frontend

# Last 100 lines
pm2 logs --lines 100
```

### Restart Apps
```bash
# Restart all
pm2 restart ecosystem.config.js

# Restart specific app
pm2 restart ti-agent-backend
pm2 restart ti-agent-frontend

# Reload (0-downtime)
pm2 reload ecosystem.config.js
```

### Stop/Start Apps
```bash
# Stop all
pm2 stop ecosystem.config.js

# Start all
pm2 start ecosystem.config.js

# Delete all
pm2 delete ecosystem.config.js
```

### Monitor
```bash
# Real-time monitoring
pm2 monit

# Memory/CPU usage
pm2 list
```

## Updating the App

### Quick Update (Code Only)
```bash
cd /home/vagarwal/tiagent/tiagent
git pull origin main
pm2 restart ecosystem.config.js
```

### Full Update (Dependencies + Build)
```bash
cd /home/vagarwal/tiagent/tiagent
git pull origin main
./deploy-pm2.sh
```

## Nginx Configuration

The nginx config at `/etc/nginx/sites-available/tiagent.theagarwals.com`:
- Serves frontend static files from `frontend/build/`
- Proxies `/api/` to backend on port 8000
- Proxies `/ws/` for WebSocket connections

### Reload Nginx
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### View Nginx Logs
```bash
# Access logs
sudo tail -f /var/log/nginx/access.log

# Error logs
sudo tail -f /var/log/nginx/error.log
```

## Troubleshooting

### Backend Not Starting
```bash
# Check PM2 logs
pm2 logs ti-agent-backend --lines 50

# Check if venv is activated
which python
# Should be: /home/vagarwal/tiagent/tiagent/venv/bin/python

# Test manually
source venv/bin/activate
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### Frontend Not Starting
```bash
# Check PM2 logs
pm2 logs ti-agent-frontend --lines 50

# Rebuild frontend
cd frontend
npm install
npm run build
cd ..
pm2 restart ti-agent-frontend
```

### 502 Bad Gateway
- Backend not running: `pm2 restart ti-agent-backend`
- Port conflict: `sudo lsof -i :8000` (kill conflicting process)
- Check nginx error log: `sudo tail -f /var/log/nginx/error.log`

### WebSocket Connection Failed
- Ensure backend is running: `pm2 status`
- Check nginx config has WebSocket headers
- Verify firewall allows connections

### Out of Memory
```bash
# Check memory usage
free -h
pm2 list

# Restart high-memory app
pm2 restart ti-agent-backend
```

## File Structure

```
/home/vagarwal/tiagent/tiagent/
├── backend/
│   ├── agent/
│   ├── app/
│   ├── ingestion/
│   └── vectorstore/
├── frontend/
│   ├── build/          # Production build
│   └── src/
├── venv/               # Python virtual environment
├── .env                # Environment variables (create manually)
├── ecosystem.config.js # PM2 configuration
├── deploy-pm2.sh       # Deployment script
├── nginx-tichat.conf   # Nginx configuration
└── requirements.txt

/home/vagarwal/tiagent/logs/
├── backend-error.log
├── backend-out.log
├── frontend-error.log
└── frontend-out.log
```

## URLs

- **Frontend**: https://tiagent.theagarwals.com
- **Backend API**: https://tiagent.theagarwals.com/api/
- **WebSocket**: wss://tiagent.theagarwals.com/ws/
- **Health Check**: https://tiagent.theagarwals.com/api/health

## PM2 Ecosystem File

Located at `ecosystem.config.js`:
- **ti-agent-backend**: Runs uvicorn on port 8000
- **ti-agent-frontend**: Serves production build

Auto-starts on server reboot (after `pm2 startup`).
