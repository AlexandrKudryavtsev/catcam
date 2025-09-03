package usecase

import (
	"context"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
)

type (
	Camera interface {
		HandleEvent(ctx context.Context, event entity.CameraEvent) error
		GetEvents(ctx context.Context) ([]entity.CameraEvent, error)
		StartStream(ctx context.Context, url string) error
		StopStream(ctx context.Context) error
		UpdateStatus(ctx context.Context, online bool) error
		GetStatus(ctx context.Context) (bool, error)
	}

	MQTTPublisher interface {
		Publish(topic string, payload interface{}) error
	}

	MQTTSubscriber interface {
		Subscribe(topic string, handler func(topic string, payload []byte)) error
	}

	MQTTClient interface {
		MQTTPublisher
		MQTTSubscriber
		Close()
	}

	StreamHandler interface {
		GenerateStreamURL(ctx context.Context) (string, error)
		TerminateStream(ctx context.Context, streamID string) error
	}
)
