package main

import (
	"crypto/md5"
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
)

//go:embed assets/favicon.png
var Favicon embed.FS

//go:embed assets/index.html
var HomePage []byte

//go:embed assets
var Assets embed.FS

var GLOBAL_API_URL = "http://127.0.0.1:5000"
var GLOBAL_CONSOLE_URL = "http://127.0.0.1:8090"

func init() {
	apiUrlFromEnv := strings.ToLower(strings.TrimSpace(os.Getenv("API_URL")))
	if apiUrlFromEnv != "" {
		GLOBAL_API_URL = apiUrlFromEnv
		fmt.Println("Use the user to specify the server:", GLOBAL_API_URL)
	} else {
		fmt.Println("`API_URL` not detected, use default:", GLOBAL_API_URL)
	}
}

func proxyAPI(c *gin.Context) {
	remote, err := url.Parse(GLOBAL_API_URL)
	if err != nil {
		panic(err)
	}

	proxy := httputil.NewSingleHostReverseProxy(remote)
	proxy.Director = func(req *http.Request) {
		req.Header = c.Request.Header
		req.Host = remote.Host
		req.URL.Scheme = remote.Scheme
		req.URL.Host = remote.Host
		req.URL.Path = c.Param("proxyPath")
	}

	proxy.ServeHTTP(c.Writer, c.Request)
}

func proxyConsole(c *gin.Context) {
	remote, err := url.Parse(GLOBAL_CONSOLE_URL)
	if err != nil {
		panic(err)
	}

	proxy := httputil.NewSingleHostReverseProxy(remote)
	proxy.Director = func(req *http.Request) {
		req.Header = c.Request.Header
		req.Host = remote.Host
		req.URL.Scheme = remote.Scheme
		req.URL.Host = remote.Host
		req.URL.Path = c.Param("proxyPath")
	}

	proxy.ServeHTTP(c.Writer, c.Request)
}

// ViewHandler support dist handler from UI
// https://github.com/gin-gonic/gin/issues/1222
func optimizeResourceCacheTime() gin.HandlerFunc {
	data := []byte(time.Now().String())
	/* #nosec */
	etag := fmt.Sprintf("W/%x", md5.Sum(data))
	return func(c *gin.Context) {
		if strings.HasPrefix(c.Request.RequestURI, "/assets/") ||
			strings.HasPrefix(c.Request.RequestURI, "/favicon.png") {
			c.Header("Cache-Control", "public, max-age=31536000")
			c.Header("ETag", etag)

			if match := c.GetHeader("If-None-Match"); match != "" {
				if strings.Contains(match, etag) {
					c.Status(http.StatusNotModified)
					return
				}
			}
		}
	}
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.Use(gzip.Gzip(gzip.DefaultCompression))
	r.Use(optimizeResourceCacheTime())

	r.Any("/", func(c *gin.Context) {
		c.Data(http.StatusOK, "text/html; charset=utf-8", HomePage)
	})

	favicon, _ := fs.Sub(Favicon, "assets")
	r.Any("/favicon.png", func(c *gin.Context) {
		c.FileFromFS("favicon.png", http.FS(favicon))
	})

	static, _ := fs.Sub(Assets, "assets/static")
	r.StaticFS("/static", http.FS(static))

	r.GET("/env-config.js", func(c *gin.Context) {
		c.Data(http.StatusOK, "application/javascript; charset=utf-8", []byte(`window._env_ = {API_URL: "/api"}`))
		c.Abort()
	})
	r.Any("/api/*proxyPath", proxyAPI)
	r.Any("/console/*proxyPath", proxyConsole)
	r.Run(":80")
}
