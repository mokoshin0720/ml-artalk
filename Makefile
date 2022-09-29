init:
	docker-compose build
	docker-compose up -d

run:
	# docker-compose exec artalk python src/construct_data/classify_abstruct_or_concrete.py
	docker-compose exec artalk python src/construct_data/extract_object.py

down:
	docker-compose down