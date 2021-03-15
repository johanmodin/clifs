#!/bin/bash

# Collect static files
#echo "Collect static files"
#python manage.py clear_cache
python manage.py collectstatic --noinput

python manage.py makemigrations

# Apply database migrations
echo "Apply database migrations"
python manage.py migrate --run-syncdb

# Start server
echo "Starting server"
python manage.py runserver 0.0.0.0:8000
