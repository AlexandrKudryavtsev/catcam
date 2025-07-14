# GoMovieSearch

Required:

1) Docker
2) docker-compose
3) Make

Commands:

1) Docker Compose
    - `compose-up` run docker-compose with Elasticsearch and app
    - `compose-devs` run docker-compose with Elasticsearch
    - `compose-down` stop and remove containers
    - `compose-logs` show logs for all services
    - `remove-volume` remove docker volume

2) Application
    - `run-app` run application locally (requires Elasticsearch)

3) Testing
    - `test` run unit tests
    - `integration-test` run integration tests

3) Development Tools
    - `generate-docs` generate API docs
    - `mock` generate mocks

4) Build
    - `build` build the application

5) Dependencies
    - `install-deps` install development dependencies

Documentation: `/api/docs/index.html`

Structure

```
├── Dockerfile
├── Makefile
├── .env               # for frequently changed or private variables
├── README.md
├── cmd
│   └── app
│       └── main.go    # app launch point 
├── config
│   ├── config.go      # parsing configs
│   └── config.yml     # for other variables
├── docker-compose.yml
├── go.mod
├── go.sum
├── internal           # only the business logic of app
│   ├── app
│   │   ├── app.go     # connects all parts, launching app
│   │   └── migrate.go
│   ├── controller     # all kinds of controllers for HTTP, GRPC, RabbitMQ, etc.
│   ├── entity         # app entities
│   └── usecase        # controller processing
├── migrations         # migrations files
├── scripts            # javascript parser of kinopoisk.ru
└── pkg                # only auxiliary code
    ├── httpserver
    ├── logger
    └── elastic
```
