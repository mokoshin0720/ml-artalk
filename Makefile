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

nohup:
	docker-compose -f docker-compose.${env}.yml exec -d artalk python -B ${filename}

down:
	docker-compose -f docker-compose.${env}.yml down

install:
	docker-compose -f docker-compose.${env}.yml exec artalk pip install -r requirements.txt

top:
	docker-compose -f docker-compose.${env}.yml top

bash:
	docker-compose -f docker-compose.prod.yml exec artalk bash

shell-cmd:
	docker-compose -f docker-compose.${env}.yml exec bash ${filename}