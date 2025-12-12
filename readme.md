# WSL venv
source .venv/bin/activate
python -m pip freeze > requirements.txt

docker-compose -f ./iac/compose.yml  --env-file ./iac/.env up