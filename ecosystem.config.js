module.exports = {
  apps: [
    {
      name: 'ti-agent-backend',
      cwd: '/home/vagarwal/tiagent/tiagent',
      script: 'venv/bin/uvicorn',
      args: 'backend.app.main:app --host 127.0.0.1 --port 8000',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'production',
        PATH: '/home/vagarwal/tiagent/tiagent/venv/bin:/usr/local/bin:/usr/bin:/bin'
      },
      error_file: '/home/vagarwal/tiagent/logs/backend-error.log',
      out_file: '/home/vagarwal/tiagent/logs/backend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
    // Frontend served as static files by nginx - no PM2 process needed
  ]
};
