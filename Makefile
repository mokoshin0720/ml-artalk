filename ?=
env = prod
ifeq ($(env), local)
	env=local
endif

init:
	docker-compose -f docker-compose.${env}.yml build
	docker-compose -f docker-compose.${env}.yml up -d

run:
	docker-compose -f docker-compose.${env}.yml exec artalk python -B ${filename}

down:
	docker-compose -f docker-compose.${env}.yml down

install:
	docker-compose -f docker-compose.${env}.yml exec artalk pip install -r requirements.txt