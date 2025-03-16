docker compose -f docker-compose.deploy.yaml up -d --build
docker exec innhabit_server python3 manage.py migrate
source venv/bin/activate
python3 manage.py tailwind build
python3 manage.py collectstatic --noinput
