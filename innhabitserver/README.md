# INNHABIT Server

## Prerequisites
* python 3.10 or newer with pip and venv
* docker and docker compose
* npm

## Initial setup
```sh
cp example.env .env
sudo apt install libpq-dev python3-dev
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
docker compose up -d
python3 manage.py migrate
python3 manage.py tailwind install
python3 manage.py runserver
```

## Running the server
```sh
source venv/bin/activate
docker compose up -d
python3 manage.py runserver
```


## When adding tailwind styles
```sh
python3 manage.py tailwind start
```
