// @title           AlexandrKudryavtsev/catcam
// @version         1.0
// @description     Backend for IoT and CV project

// @BasePath  /api
// @schemes https http

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
package main

import (
	"log"

	"github.com/AlexandrKudryavtsev/catcam/config"
	"github.com/AlexandrKudryavtsev/catcam/internal/app"
)

func main() {
	cfg, err := config.NewConfig()
	if err != nil {
		log.Fatalf("can't init config: %s", err)
	}

	app.Run(cfg)
}
