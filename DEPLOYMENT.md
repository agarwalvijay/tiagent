# TI Agent Deployment Guide

Deploy TI Agent to **tiagent.theagarwals.com** on GCP

## Prerequisites

### 1. DNS Setup
Point `tiagent.theagarwals.com` A record to your GCP server IP

### 2. Server Requirements
- Ubuntu/Debian Linux
- Python 3.9+
- Node.js 16+
- Nginx installed
- Sufficient disk space for vector database (~500MB+)

### 3. Server Access
```bash
ssh vagarwal@tiagent.theagarwals.com
```

## First-Time Deployment

### Step 1: Build Frontend Locally
```bash
cd frontend
npm install
npm run build
cd ..
```

### Step 2: Sync to Server
```bash
./sync-to-server.sh
```

This will sync all files except:
- Source code (frontend/src, frontend/public)
- Dependencies (node_modules, venv)
- Vector database (backend/vectorstore/)
- Local .env files
- Datasheets

### Step 3: Set Up Environment on Server
SSH into the server:
```bash
ssh vagarwal@tiagent.theagarwals.com
cd /home/vagarwal/ti-agent
```

Create `.env` file with your credentials:
```bash
nano .env
```

Add your environment variables:
```env
# LLM Provider (openai, groq, deepseek, google)
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI (if using)
# OPENAI_API_KEY=your_openai_key

# Embeddings
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_for_embeddings

# Backend
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Frontend
FRONTEND_URL=http://tiagent.theagarwals.com
```

### Step 4: Deploy
Run the deployment script on the server:
```bash
./deploy.sh
```

This will:
1. Create Python virtual environment
2. Install Python dependencies
3. Install Node.js dependencies (if needed)
4. Build frontend
5. Set up systemd service
6. Configure nginx
7. Start the application

### Step 5: Set Up SSL (HTTPS)
Install certbot:
```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

Get SSL certificate:
```bash
sudo certbot --nginx -d tiagent.theagarwals.com
```

Follow the prompts. Certbot will:
- Automatically modify nginx config to use SSL
- Set up auto-renewal

Test auto-renewal:
```bash
sudo certbot renew --dry-run
```

### Step 6: Ingest Data
Upload your datasheets and run ingestion:
```bash
# Upload Datasheets folder (if not already synced)
rsync -avz --progress Datasheets/ vagarwal@tiagent.theagarwals.com:/home/vagarwal/ti-agent/Datasheets/

# SSH into server
ssh vagarwal@tiagent.theagarwals.com
cd /home/vagarwal/ti-agent

# Run ingestion
source venv/bin/activate
python -m backend.ingestion.ingest_pdfs
```

## Updating the Application

### Option 1: Quick Update (Code Only)
```bash
# On local machine
./sync-to-server.sh

# On server
ssh vagarwal@tiagent.theagarwals.com
cd /home/vagarwal/ti-agent
sudo systemctl restart ti-agent
```

### Option 2: Full Redeploy (Code + Dependencies)
```bash
# On local machine - rebuild frontend
cd frontend && npm run build && cd ..
./sync-to-server.sh

# On server
ssh vagarwal@tiagent.theagarwals.com
cd /home/vagarwal/ti-agent
./deploy.sh
```

## Monitoring

### Check Service Status
```bash
sudo systemctl status ti-agent
```

### View Logs
```bash
# Backend logs
sudo journalctl -u ti-agent -f

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Restart Services
```bash
# Restart backend
sudo systemctl restart ti-agent

# Restart nginx
sudo systemctl restart nginx
```

## Troubleshooting

### Backend Not Starting
1. Check logs: `sudo journalctl -u ti-agent -n 50`
2. Check if port 8000 is in use: `sudo lsof -i :8000`
3. Verify Python environment: `/home/vagarwal/ti-agent/venv/bin/python --version`
4. Check dependencies: `source venv/bin/activate && pip list`

### Nginx 502 Bad Gateway
- Backend service is not running: `sudo systemctl start ti-agent`
- Backend crashed: Check logs with `sudo journalctl -u ti-agent`
- Port mismatch: Verify backend runs on 127.0.0.1:8000

### WebSocket Connection Failed
- Check nginx config has WebSocket headers
- Verify `/ws/` location block in nginx config
- Check firewall allows WebSocket connections

### Out of Memory
- Check memory usage: `free -h`
- Consider adding swap: `sudo fallocate -l 4G /swapfile`
- Reduce number of workers in uvicorn

## File Structure on Server

```
/home/vagarwal/ti-agent/
├── backend/
│   ├── agent/
│   ├── app/
│   ├── ingestion/
│   └── vectorstore/      # Created during ingestion
├── frontend/
│   └── build/            # Production build
├── venv/                 # Python virtual environment
├── .env                  # Environment variables (create manually)
├── deploy.sh
├── ti-agent.service
├── nginx-tichat.conf
└── requirements.txt
```

## URLs

- **Frontend**: https://tiagent.theagarwals.com
- **Backend API**: https://tiagent.theagarwals.com/api/
- **WebSocket**: wss://tiagent.theagarwals.com/ws/
- **Health Check**: https://tiagent.theagarwals.com/api/health

## Security Checklist

- [ ] `.env` file has correct permissions: `chmod 600 .env`
- [ ] SSL certificate installed and auto-renewing
- [ ] Firewall configured (UFW): Allow 80, 443, 22
- [ ] Backend only listens on 127.0.0.1 (not public)
- [ ] Regular system updates: `sudo apt-get update && sudo apt-get upgrade`
- [ ] API keys rotated regularly
- [ ] Logs monitored for errors/attacks

## Performance Tips

1. **Enable nginx caching** for static assets (already configured)
2. **Use Redis** for session storage instead of in-memory
3. **Add gunicorn** with multiple workers:
   ```bash
   gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
   ```
4. **Monitor disk space** for logs and vector DB
5. **Set up CloudFlare** for CDN and DDoS protection
