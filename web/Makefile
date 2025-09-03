include .env
export

LOCAL_BIN:=$(CURDIR)/bin
PATH:=$(LOCAL_BIN):$(PATH)

.PHONY: help

help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Docker Compose

compose-up: ### Run docker-compose with devs and app
	docker-compose up --build -d app mqtt5

compose-devs: ### Run docker-compose with devs
	docker-compose up --build -d mqtt5

compose-down: ### Stop and remove containers
	docker-compose down --remove-orphans

compose-logs: ### Show logs for all services
	docker-compose logs -f

remove-volume: ### Remove docker volume
	docker volume rm gomoviesearch_es-data

##@ Application

run-app: ### Run application locally (requires Elasticsearch)
	go run ./cmd/app

##@ Testing

test: ### Run unit tests
	go test -v -cover -race ./internal/...
.PHONY: test

integration-test: ### Run integration tests
	go clean -testcache && go test -v ./integration-test/...
.PHONY: integration-test

##@ Development Tools

generate-docs: ### Generate API docs
	./bin/swag init -g cmd/app/main.go
.PHONY: generate-docs

mock: ### Generate mocks
	mockgen -source ./internal/usecase/interfaces.go -package usecase_test > ./internal/usecase/mocks_test.go
.PHONY: mock

##@ Build

build: ### Build the application
	go build -o $(LOCAL_BIN)/app ./cmd/app

##@ Dependencies

install-deps: ### Install development dependencies
	GOBIN=$(LOCAL_BIN) go install github.com/swaggo/swag/cmd/swag@latest
	GOBIN=$(LOCAL_BIN) go install github.com/golang/mock/mockgen@latest
.PHONY: install-deps