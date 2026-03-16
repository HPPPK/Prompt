process.env.HOST = process.env.HOST && process.env.HOST.trim()
  ? process.env.HOST.trim()
  : '127.0.0.1';

if (!process.env.DANGEROUSLY_DISABLE_HOST_CHECK) {
  process.env.DANGEROUSLY_DISABLE_HOST_CHECK = 'true';
}

require('react-scripts/scripts/start');
