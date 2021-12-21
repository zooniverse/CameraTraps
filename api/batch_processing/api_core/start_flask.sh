#!/bin/sh

LISTEN_PORT=${LISTEN_PORT:=80}

if [ "$FLASK_ENV" = "production" ]; then
  echo Starting production server
  exec gunicorn -b 0:$LISTEN_PORT -w 1 --threads 4 -t 60 --access-logfile - --capture-output server:app
else
  echo Starting development server
  exec flask run -p 8080 --eager-loading --no-reload
fi

