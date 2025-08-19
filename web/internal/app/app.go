package app

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/AlexandrKudryavtsev/catcam/config"
	v1 "github.com/AlexandrKudryavtsev/catcam/internal/controller/http/v1"
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase"
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase/repository"
	"github.com/AlexandrKudryavtsev/catcam/pkg/httpserver"
	"github.com/AlexandrKudryavtsev/catcam/pkg/logger"
	"github.com/AlexandrKudryavtsev/catcam/pkg/mqtt"
	"github.com/gin-gonic/gin"
)

func Run(cfg *config.Config) {
	logger, err := logger.New(cfg.Log.Level, cfg.Log.Destination)
	if err != nil {
		log.Fatalf("can't init logger: %s", err)
	}
	logger.Info("logger initialized")

	// 1. Инициализация репозитория
	repo := repository.NewMemoryRepository()
	uc := usecase.NewCameraUseCase(repo)
	logger.Info("usecase layer initialized")

	// 2. Инициализация MQTT клиента
	mqttClient, err := mqtt.NewClient("tcp://mqtt5:1883", "catcam-server")
	if err != nil {
		logger.Fatal(fmt.Errorf("MQTT connection error: %w", err))
	}
	defer mqttClient.Close()
	logger.Info("MQTT client connected successfully")

	// 3. Инициализация MQTT обработчиков
	mqttHandler := NewMQTTHandler(uc, *logger)
	if err := mqttClient.Subscribe("catcam/events", mqttHandler.HandleEvents); err != nil {
		logger.Error(fmt.Errorf("failed to subscribe to events: %w", err))
	}
	if err := mqttClient.Subscribe("catcam/status", mqttHandler.HandleStatus); err != nil {
		logger.Error(fmt.Errorf("failed to subscribe to status: %w", err))
	}
	logger.Info("MQTT subscriptions established")

	// 4. Инициализация HTTP сервера
	handler := gin.New()
	v1.NewRouter(handler, uc, logger)

	httpServer := httpserver.New(handler, httpserver.Port(cfg.HTTP.Port))
	logger.Info(fmt.Sprintf("HTTP server started on port %s", cfg.HTTP.Port))

	// 5. Ожидание сигналов завершения
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt, syscall.SIGTERM)

	select {
	case s := <-interrupt:
		logger.Info("app - Run - signal: " + s.String())
	case err = <-httpServer.Notify():
		logger.Error(fmt.Errorf("app - Run - httpServer.Notify: %w", err))
	}

	if err := httpServer.Shutdown(); err != nil {
		logger.Error(fmt.Errorf("app - Run - httpServer.Shutdown: %w", err))
	}
}
