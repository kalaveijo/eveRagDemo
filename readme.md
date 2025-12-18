# WSL venv
source .venv/bin/activate
python -m pip freeze > requirements.txt

docker compose --profile full up