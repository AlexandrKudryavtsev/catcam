package repository

import (
	"context"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
)

type CameraRepository interface {
	SaveEvent(ctx context.Context, event entity.CameraEvent) error
	GetEvents(ctx context.Context) ([]entity.CameraEvent, error)
	SaveStreamURL(ctx context.Context, url string) error
	GetStreamURL(ctx context.Context) (string, error)
	UpdateCameraStatus(ctx context.Context, online bool) error
	GetCameraStatus(ctx context.Context) (bool, error)
}
