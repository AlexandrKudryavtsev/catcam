package usecase

import (
	"context"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase/repository"
)

type CameraUseCase struct {
	repo repository.CameraRepository
}

func NewCameraUseCase(repo repository.CameraRepository) *CameraUseCase {
	return &CameraUseCase{repo: repo}
}

func (uc *CameraUseCase) HandleEvent(ctx context.Context, event entity.CameraEvent) error {
	return uc.repo.SaveEvent(ctx, event)
}

func (uc *CameraUseCase) GetEvents(ctx context.Context) ([]entity.CameraEvent, error) {
	return uc.repo.GetEvents(ctx)
}

func (uc *CameraUseCase) StartStream(ctx context.Context, url string) error {
	return uc.repo.SaveStreamURL(ctx, url)
}

func (uc *CameraUseCase) StopStream(ctx context.Context) error {
	return uc.repo.SaveStreamURL(ctx, "")
}

func (uc *CameraUseCase) UpdateStatus(ctx context.Context, online bool) error {
	return uc.repo.UpdateCameraStatus(ctx, online)
}

func (uc *CameraUseCase) GetStatus(ctx context.Context) (bool, error) {
	return uc.repo.GetCameraStatus(ctx)
}
