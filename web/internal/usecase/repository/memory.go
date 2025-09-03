package repository

import (
	"context"
	"sync"
	"time"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
)

type MemoryRepository struct {
	mu           sync.RWMutex
	events       []entity.CameraEvent
	streamURL    string
	cameraStatus bool
}

func NewMemoryRepository() *MemoryRepository {
	return &MemoryRepository{}
}

func (r *MemoryRepository) SaveEvent(ctx context.Context, event entity.CameraEvent) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	event.Timestamp = time.Now()
	r.events = append(r.events, event)
	return nil
}

func (r *MemoryRepository) GetEvents(ctx context.Context) ([]entity.CameraEvent, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.events, nil
}

func (r *MemoryRepository) SaveStreamURL(ctx context.Context, url string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.streamURL = url
	return nil
}

func (r *MemoryRepository) GetStreamURL(ctx context.Context) (string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.streamURL, nil
}

func (r *MemoryRepository) UpdateCameraStatus(ctx context.Context, online bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.cameraStatus = online
	return nil
}

func (r *MemoryRepository) GetCameraStatus(ctx context.Context) (bool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.cameraStatus, nil
}
