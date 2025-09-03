package v1

import (
	"github.com/AlexandrKudryavtsev/catcam/internal/usecase"
	"github.com/AlexandrKudryavtsev/catcam/pkg/logger"
	"github.com/gin-gonic/gin"
)

func NewRouter(handler *gin.Engine, uc usecase.Camera, l logger.Interface) {
	handler.Use(gin.Logger())
	handler.Use(gin.Recovery())

	router := handler.Group("/api")

	newCommonRoutes(router)
	newCameraRoutes(router, uc, l)
}
