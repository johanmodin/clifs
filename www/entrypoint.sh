#!/bin/bash

# Collect static files
#echo "Collect static files"
#python manage.py clear_cache
python3 manage.py collectstatic --noinput

python3 manage.py makemigrations

# Apply database migrations
echo "Apply database migrations"
python3 manage.py migrate --run-syncdb

# Start server
echo "Starting server"
python3 manage.py runserver 0.0.0.0:8000
