package entity

import "time"

type EventType string

const (
	EventTypeCatDetected EventType = "cat_detected"
)

type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

type CameraEvent struct {
	EventType   EventType   `json:"event_type"`
	Timestamp   time.Time   `json:"timestamp"`
	BoundingBox BoundingBox `json:"bounding_box"`
}

type ControlCommand struct {
	Command string `json:"command"`
}

type StreamInfo struct {
	URL string `json:"url"`
}

type CameraStatus struct {
	Online bool `json:"online"`
}
