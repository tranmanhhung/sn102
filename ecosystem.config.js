module.exports = {
  apps: [
    {
      name: 'optimized-miner',
      script: 'run_optimized_miner.py',
      interpreter: 'python3',
      args: '--config mid_range --model balanced',
      cwd: '/Users/hungtranmanh/Projects/BetterTherapy-Subnet',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '4G',
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: '/Users/hungtranmanh/Projects/BetterTherapy-Subnet',
        TOKENIZERS_PARALLELISM: 'false',
        HF_HUB_DISABLE_PROGRESS_BARS: '1'
      },
      error_file: './logs/optimized-miner-error.log',
      out_file: './logs/optimized-miner-out.log',
      log_file: './logs/optimized-miner-combined.log',
      time: true,
      max_restarts: 10,
      min_uptime: '10s',
      kill_timeout: 30000
    },
    {
      name: 'standard-miner',
      script: 'neurons/miner.py',
      interpreter: 'python3',
      args: '--netuid 34 --subtensor.network finney --logging.debug',
      cwd: '/Users/hungtranmanh/Projects/BetterTherapy-Subnet',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '3G',
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: '/Users/hungtranmanh/Projects/BetterTherapy-Subnet'
      },
      error_file: './logs/standard-miner-error.log',
      out_file: './logs/standard-miner-out.log',
      log_file: './logs/standard-miner-combined.log',
      time: true,
      max_restarts: 10,
      min_uptime: '10s',
      kill_timeout: 30000
    }
  ]
}
