#!/usr/bin/env bash
set -e

# Change into backend folder and start the Flask app via gunicorn (Railway / Railpack expects a start script)
cd backend

# Default PORT if not provided by the platform
: "${PORT:=5000}"

if command -v gunicorn >/dev/null 2>&1; then
  exec gunicorn main:app --bind 0.0.0.0:${PORT} --workers 1
else
  echo "gunicorn not found, running with python (slower)."
  exec python main.py
fi
