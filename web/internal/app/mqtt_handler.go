package app

import (
	"context"
	"encoding/json"
	"time"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase"
	"github.com/AlexandrKudryavtsev/catcam/pkg/logger"
)

type MQTTHandler struct {
	uc     usecase.Camera
	logger logger.Logger
}

func NewMQTTHandler(uc usecase.Camera, logger logger.Logger) *MQTTHandler {
	return &MQTTHandler{uc: uc, logger: logger}
}

func (h *MQTTHandler) HandleEvents(topic string, payload []byte) {
	var event entity.CameraEvent
	if err := json.Unmarshal(payload, &event); err != nil {
		h.logger.Error("failed to unmarshal event: %v", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := h.uc.HandleEvent(ctx, event); err != nil {
		h.logger.Error("failed to handle event: %v", err)
	}
}

func (h *MQTTHandler) HandleStatus(topic string, payload []byte) {
	var status struct {
		Online bool `json:"online"`
	}
	if err := json.Unmarshal(payload, &status); err != nil {
		h.logger.Error("failed to unmarshal status: %v", err)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := h.uc.UpdateStatus(ctx, status.Online); err != nil {
		h.logger.Error("failed to update status: %v", err)
	}
}
