init:
	docker-compose build
	docker-compose up -d

run:
	docker-compose exec artalk python src/extract_object.py

down:
	docker-compose down