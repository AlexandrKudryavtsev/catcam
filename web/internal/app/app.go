package app

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/AlexandrKudryavtsev/catcam/config"
	v1 "github.com/AlexandrKudryavtsev/catcam/internal/controller/http/v1"
	"github.com/AlexandrKudryavtsev/catcam/pkg/httpserver"
	"github.com/AlexandrKudryavtsev/catcam/pkg/logger"
	"github.com/gin-gonic/gin"
)

func Run(cfg *config.Config) {
	logger, err := logger.New(cfg.Log.Level, cfg.Log.Destination)
	if err != nil {
		log.Fatalf("can't init logger: %s", err)
	}
	logger.Info("logger init")

	handler := gin.New()
	v1.NewRouter(handler, logger)

	httpServer := httpserver.New(handler, httpserver.Port(cfg.HTTP.Port))

	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt, syscall.SIGTERM)

	select {
	case s := <-interrupt:
		logger.Info("signal: " + s.String())
	case err = <-httpServer.Notify():
		logger.Error(fmt.Errorf("app - Run - httpServer.Notify: %w", err))
	}

	if err := httpServer.Shutdown(); err != nil {
		logger.Error(fmt.Errorf("app - Run - httpServer.Shutdown: %w", err))
	}
}
