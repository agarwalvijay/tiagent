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
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PATH: '/home/vagarwal/tiagent/tiagent/venv/bin:/usr/local/bin:/usr/bin:/bin'
      },
      error_file: '/home/vagarwal/tiagent/logs/backend-error.log',
      out_file: '/home/vagarwal/tiagent/logs/backend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'ti-agent-frontend',
      cwd: '/home/vagarwal/tiagent/tiagent/frontend',
      script: 'npm',
      args: 'start',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '512M',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
        BROWSER: 'none'
      },
      error_file: '/home/vagarwal/tiagent/logs/frontend-error.log',
      out_file: '/home/vagarwal/tiagent/logs/frontend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
