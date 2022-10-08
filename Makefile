init:
	docker-compose build
	docker-compose up -d

run:
	docker-compose exec artalk python -B src/experiment/dataset.py

down:
	docker-compose down

install:
	docker-compose exec artalk pip install -r requirements.txt