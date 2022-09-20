package main

import (
	"embed"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
)

//go:embed assets
var assets embed.FS

var GLOBAL_API_URL = "http://192.168.1.85:5000"

func getConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "application/javascript")
	io.WriteString(w, `window._env_ = {API_URL: "`+GLOBAL_API_URL+`"}`)
}

func init() {
	apiUrlFromEnv := strings.ToLower(strings.TrimSpace(os.Getenv("API_URL")))
	if strings.HasPrefix(apiUrlFromEnv, "http://") || strings.HasPrefix(apiUrlFromEnv, "https://") {
		GLOBAL_API_URL = apiUrlFromEnv
		fmt.Println("Use the user to specify the server:", GLOBAL_API_URL)
	} else {
		fmt.Println("`API_URL` not detected, use default:", GLOBAL_API_URL)
	}
}

func main() {
	mutex := http.NewServeMux()
	md, _ := fs.Sub(assets, "assets")
	mutex.Handle("/", http.FileServer(http.FS(md)))
	mutex.HandleFunc("/env-config.js", getConfig)

	err := http.ListenAndServe(":3080", mutex)
	if err != nil {
		log.Fatal(err)
	}
}
