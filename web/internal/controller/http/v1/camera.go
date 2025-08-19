package v1

import (
	"net/http"

	"github.com/AlexandrKudryavtsev/catcam/internal/entity"
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase"
	"github.com/AlexandrKudryavtsev/catcam/pkg/logger"
	"github.com/gin-gonic/gin"
)

type cameraRoutes struct {
	uc usecase.Camera
	l  logger.Interface
}

func newCameraRoutes(handler *gin.RouterGroup, uc usecase.Camera, l logger.Interface) {
	r := &cameraRoutes{uc, l}

	h := handler.Group("/camera")
	{
		h.GET("/events", r.getEvents)
		h.POST("/stream/start", r.startStream)
		h.POST("/stream/stop", r.stopStream)
		h.GET("/status", r.getStatus)
	}
}

type getEventsResponse struct {
	Events []entity.CameraEvent `json:"events"`
}

// @Summary     Get camera events
// @Description Get all detected events
// @ID          get-events
// @Tags  	    camera
// @Accept      json
// @Produce     json
// @Success     200 {object} getEventsResponse
// @Failure     500 {object} response
// @Router      /camera/events [get]
func (r *cameraRoutes) getEvents(c *gin.Context) {
	events, err := r.uc.GetEvents(c.Request.Context())
	if err != nil {
		errorResponse(c, http.StatusInternalServerError, "failed to get events")
		return
	}

	c.JSON(http.StatusOK, getEventsResponse{Events: events})
}

type startStreamRequest struct {
	URL string `json:"url" binding:"required"`
}

// @Summary     Start video stream
// @Description Start video stream from camera
// @ID          start-stream
// @Tags  	    camera
// @Accept      json
// @Produce     json
// @Param       request body startStreamRequest true "Stream URL"
// @Success     200
// @Failure     400 {object} response
// @Failure     500 {object} response
// @Router      /camera/stream/start [post]
func (r *cameraRoutes) startStream(c *gin.Context) {
	var req startStreamRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		errorResponse(c, http.StatusBadRequest, "invalid request body")
		return
	}

	if err := r.uc.StartStream(c.Request.Context(), req.URL); err != nil {
		errorResponse(c, http.StatusInternalServerError, "failed to start stream")
		return
	}

	c.Status(http.StatusOK)
}

// @Summary     Stop video stream
// @Description Stop video stream from camera
// @ID          stop-stream
// @Tags  	    camera
// @Accept      json
// @Produce     json
// @Success     200
// @Failure     500 {object} response
// @Router      /camera/stream/stop [post]
func (r *cameraRoutes) stopStream(c *gin.Context) {
	if err := r.uc.StopStream(c.Request.Context()); err != nil {
		errorResponse(c, http.StatusInternalServerError, "failed to stop stream")
		return
	}

	c.Status(http.StatusOK)
}

type getStatusResponse struct {
	Online bool `json:"online"`
}

// @Summary     Get camera status
// @Description Get camera online status
// @ID          get-status
// @Tags  	    camera
// @Accept      json
// @Produce     json
// @Success     200 {object} getStatusResponse
// @Failure     500 {object} response
// @Router      /camera/status [get]
func (r *cameraRoutes) getStatus(c *gin.Context) {
	status, err := r.uc.GetStatus(c.Request.Context())
	if err != nil {
		errorResponse(c, http.StatusInternalServerError, "failed to get status")
		return
	}

	c.JSON(http.StatusOK, getStatusResponse{Online: status})
}
